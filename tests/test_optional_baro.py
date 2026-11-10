"""The barometer dependency: values never reach the model; timestamps only bound the grid.

These tests were written BEFORE making the barometer optional and pin the behaviour that makes the
change safe. Three-file uploads still require real barometer data.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from src import inference_service as inf
from src.preprocessing_adapter import PreprocessingError, align_uploaded_flight_detailed, baro_timestamp_scaffold
from src.upload import load_individual_csvs
from tests.conftest import _make_att, _make_baro, _make_imu

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = str(ROOT / "uav_model")  # the real deployed artifacts


def run(frames, meta=None):
    inf.clear_caches()
    try:
        return inf.analyze_flight(frames, flight_id="x", model_dir=MODEL, flight_meta=meta)
    finally:
        inf.clear_caches()


def csv(d):
    b = io.BytesIO()
    d.to_csv(b, index=False)
    return b.getvalue()


@pytest.fixture()
def base():
    return _make_imu(500), _make_att(200), _make_baro(200)


def test_scores_do_not_depend_on_barometer_values(base):
    imu, att, baro = base
    ref = run({"IMU": imu, "ATT": att, "BARO": baro})
    for alt in (baro.assign(Alt=1e6, Press=5.0, Temp=-300.0), baro.assign(Alt=np.nan, Press=np.nan, Temp=np.nan),
                baro.assign(Alt=baro.Alt[::-1].to_numpy())):
        got = run({"IMU": imu, "ATT": att, "BARO": alt})
        assert got.ok and np.array_equal(ref.windows.anomaly_score.values, got.windows.anomaly_score.values)
        assert np.array_equal(ref.windows.is_anomaly.values, got.windows.is_anomaly.values)
        assert ref.windows[["t_start", "t_end", "session_id"]].equals(got.windows[["t_start", "t_end", "session_id"]])


def test_barometer_timestamps_do_bound_the_grid(base):
    imu, att, baro = base
    full = run({"IMU": imu, "ATT": att, "BARO": baro})
    late = baro.iloc[100:].reset_index(drop=True)  # barometer starts 10 s late
    short = run({"IMU": imu, "ATT": att, "BARO": late})
    assert short.n_windows < full.n_windows  # so the scaffold must come from IMU timestamps, not be arbitrary


def test_imu_timestamp_scaffold_equals_a_barometer_with_full_coverage(base):
    imu, att, baro = base
    ref = run({"IMU": imu, "ATT": att, "BARO": baro})
    sc = baro_timestamp_scaffold(imu)
    assert sc[["Alt", "Press", "Temp"]].isna().all().all() and (sc.TimeUS.values == imu.TimeUS.values).all()
    got = run({"IMU": imu, "ATT": att}, {"baro_absent": True})
    assert got.ok and ref.n_windows == got.n_windows
    assert np.array_equal(ref.windows.anomaly_score.values, got.windows.anomaly_score.values)
    assert np.array_equal(ref.windows.is_anomaly.values, got.windows.is_anomaly.values)
    assert [e.to_dict() for e in ref.events] == [e.to_dict() for e in got.events]
    assert ref.threshold == got.threshold == 4.034964084625244
    assert any("No barometer data" in w for w in got.warnings) and got.data_quality["barometer_provided"] is False
    assert got.provenance["barometer"]["provided"] is False


def test_gaps_in_imu_still_split_sessions_without_a_barometer():
    imu = pd.concat([_make_imu(250, 1_000_000), _make_imu(250, 13_000_000)], ignore_index=True)
    att = pd.concat([_make_att(100, 1_000_000), _make_att(100, 13_000_000)], ignore_index=True)
    baro = pd.concat([_make_baro(100, 1_000_000), _make_baro(100, 13_000_000)], ignore_index=True)
    ref = run({"IMU": imu, "ATT": att, "BARO": baro})
    got = run({"IMU": imu, "ATT": att}, {"baro_absent": True})
    assert ref.windows.session_id.nunique() == got.windows.session_id.nunique() == 2
    assert np.array_equal(ref.windows.anomaly_score.values, got.windows.anomaly_score.values)


def test_missing_barometer_without_the_declaration_is_still_an_error(base):
    imu, att, _ = base
    with pytest.raises(PreprocessingError):
        align_uploaded_flight_detailed({"IMU": imu, "ATT": att}, model_dir=MODEL)
    res = run({"IMU": imu, "ATT": att})
    assert not res.ok and res.status == "error"


def test_three_file_upload_still_requires_the_barometer(base):
    imu, att, _ = base
    r = load_individual_csvs({"IMU": csv(imu), "ATT": csv(att)})
    assert not r.ok and any("BARO" in e for e in r.errors)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("UAV_MODEL_DIR", MODEL)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "flights"))
    monkeypatch.setenv("UAV_PROFILES_FILE", str(tmp_path / "none.json"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from backend import deps

    deps.get_store.cache_clear()
    inf.clear_caches()
    yield TestClient(app)
    inf.clear_caches()
    deps.get_store.cache_clear()


def test_without_a_profile_declaration_a_combined_file_still_needs_barometer_columns(client, base):
    imu, att, _ = base
    d = imu.merge(att, on="TimeUS", how="outer").sort_values("TimeUS")
    r = client.post("/api/flights/combined/auto", files={"file": ("f.csv", csv(d), "text/csv")}).json()
    assert r["stage"] == "incompatible"
    assert {m["canonical"] for m in r["assessment"]["missing"]} == {"alt", "press", "temp"}


# ---- review additions: duplicates, accounting honesty, real barometer under an optional profile ------------------------

import json
from tests import test_flt_profile as T


@pytest.fixture()
def optional_profile_env(tmp_path, monkeypatch):
    path, d = T.documented_variant(tmp_path)  # no barometer columns; profile declares them optional
    monkeypatch.setenv("UAV_PROFILES_FILE", str(path))
    monkeypatch.setenv("UAV_PROFILE_CONFIRMATIONS", str(tmp_path / "none.json"))
    return d


def test_duplicate_upload_of_a_barometer_less_flight_is_still_detected(client, optional_profile_env):
    raw = T.csv(optional_profile_env)
    first = client.post("/api/flights/combined/auto", files={"file": ("a.csv", raw, "text/csv")}).json()
    assert first["stage"] == "analyzed"
    second = client.post("/api/flights/combined/auto", files={"file": ("b.csv", raw, "text/csv")}).json()
    assert second["stage"] == "not_stored" and second["duplicate_of"] == first["flight_id"]
    assert second["duplicate_kind"] == "exact_duplicate" and len(client.get("/api/flights").json()) == 1
    # an explicit override stores a marked duplicate, as for any other upload
    third = client.post("/api/flights/combined/auto", files={"file": ("c.csv", raw, "text/csv")}, data={"allow_duplicate": "true"}).json()
    assert third["stage"] == "analyzed" and third["flight_id"] != first["flight_id"]


def test_a_barometer_less_flight_never_matches_a_three_stream_flight(base):
    from src.flight_identity import compute_identity

    imu, att, baro = base
    three = compute_identity({"IMU": imu, "ATT": att, "BARO": baro})
    two = compute_identity({"IMU": imu, "ATT": att})
    assert three["content_hash"] != two["content_hash"] and two["content_hash"]
    assert compute_identity({"IMU": imu, "ATT": att, "BARO": baro})["content_hash"] == three["content_hash"]  # stored ids unchanged


def test_data_quality_does_not_present_the_scaffold_as_barometer_rows(client, optional_profile_env):
    r = client.post("/api/flights/combined/auto", files={"file": ("a.csv", T.csv(optional_profile_env), "text/csv")}).json()
    rep = client.get(f"/api/flights/{r['flight_id']}/report").json()
    b = rep["data_quality"]["streams"]["BARO"]
    assert b["provided"] is False and b["rows_input"] == 0 and "no readings" in b["note"]
    assert rep["data_quality"]["barometer_provided"] is False
    assert rep["data_quality"]["streams"]["IMU"]["rows_input"] == len(optional_profile_env)
    from backend.deps import get_store

    assert not (get_store().flight_dir(r["flight_id"]) / "BARO.parquet").exists()  # nothing barometer-shaped is stored
    assert get_store().load_meta(r["flight_id"])["baro_absent"] is True


def test_baro_chart_does_not_fail_on_a_scaffold():
    from src.event_builder import AnomalyEvent
    from src.visualization import baro_context_figure

    t = np.arange(0, 20, 0.1)
    al = pd.DataFrame({"_t_s": t, "_session_id": 0, "alt": np.nan, "press": np.nan, "temp": np.nan})
    fig = baro_context_figure(al, AnomalyEvent(event_id="e", session_id=0, start_time_s=5, end_time_s=8))
    assert fig is not None


def test_real_barometer_columns_under_an_optional_profile_are_used_and_equal_the_three_file_flow(tmp_path, monkeypatch):
    path, d = T.documented_variant(tmp_path)
    prof = json.loads(path.read_text())
    prof["profiles"][0]["columns"].update({
        "alt_m": {"canonical": "alt", "unit": "m", "role": "signal"},
        "press_pa": {"canonical": "press", "unit": "Pa", "role": "signal"},
        "temp_c": {"canonical": "temp", "unit": "degC", "role": "signal"}})
    path.write_text(json.dumps(prof))
    monkeypatch.setenv("UAV_PROFILES_FILE", str(path))
    monkeypatch.setenv("UAV_PROFILE_CONFIRMATIONS", str(tmp_path / "none.json"))
    d = d.assign(alt_m=np.linspace(0, 1, len(d)), press_pa=101325.0, temp_c=21.0)
    from src import combined_ingest as ci

    res, _i, _m = ci.load_combined(T.csv(d), "f.csv")
    assert res.ok and not res.baro_absent and set(res.frames) == {"IMU", "ATT", "BARO"}
    t = d["t_ms"] * 1000
    sep = load_individual_csvs({
        "IMU": csv(pd.DataFrame({"TimeUS": t, "GyrX": d.gx * np.pi / 180, "GyrY": d.gy * np.pi / 180, "GyrZ": d.gz * np.pi / 180,
                                 "AccX": -d.ax, "AccY": -d.ay, "AccZ": -d.az})),
        "ATT": csv(pd.DataFrame({"TimeUS": t, "Roll": d.phi_deg, "Pitch": d.theta_deg, "Yaw": d.yaw_deg, "DesRoll": d.phi_sp, "DesPitch": d.theta_sp})),
        "BARO": csv(pd.DataFrame({"TimeUS": t, "Alt": d.alt_m, "Press": d.press_pa, "Temp": d.temp_c}))})
    a, b = run(sep.frames), run(res.frames, {"baro_absent": False})
    assert np.array_equal(a.windows.is_anomaly.values, b.windows.is_anomaly.values)
    np.testing.assert_allclose(a.windows.anomaly_score.values, b.windows.anomaly_score.values, rtol=1e-4, atol=1e-6)
    assert a.windows[["t_start", "t_end", "session_id"]].equals(b.windows[["t_start", "t_end", "session_id"]])


def test_partial_barometer_columns_are_ignored_with_a_warning(tmp_path, monkeypatch):
    path, d = T.documented_variant(tmp_path)
    prof = json.loads(path.read_text())
    prof["profiles"][0]["columns"]["alt_m"] = {"canonical": "alt", "unit": "m", "role": "signal"}
    path.write_text(json.dumps(prof))
    monkeypatch.setenv("UAV_PROFILES_FILE", str(path))
    monkeypatch.setenv("UAV_PROFILE_CONFIRMATIONS", str(tmp_path / "none.json"))
    from src import combined_ingest as ci

    res, _i, _m = ci.load_combined(T.csv(d.assign(alt_m=1.0)), "f.csv")
    assert res.ok and res.baro_absent and "BARO" not in res.frames
    assert any("Only part of the barometer" in w for w in res.warnings)


def test_a_wrong_baro_absent_flag_cannot_replace_real_barometer_data(base):
    imu, att, baro = base
    ref = run({"IMU": imu, "ATT": att, "BARO": baro})
    flagged = run({"IMU": imu, "ATT": att, "BARO": baro}, {"baro_absent": True})  # BARO present -> flag is inert
    assert np.array_equal(ref.windows.anomaly_score.values, flagged.windows.anomaly_score.values)
    assert not any("No barometer data" in w for w in flagged.warnings)
