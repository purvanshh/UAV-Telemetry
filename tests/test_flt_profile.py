"""The registered ``uav_flt_csv`` profile (33-column flight log), using the real sample files."""

from __future__ import annotations

import io
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from src import combined_ingest as ci
from src import inference_service as inf
from src import schema_llm
from src.combined_preview import build_combined_preview
from src.dataset_profiles import headers_fingerprint, load_profiles, match_profile_detailed
from src.timebase import normalize_timestamps
from src.upload import load_individual_csvs
from tests.conftest import _make_att, _make_baro, _make_imu

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "fixtures" / "data"
SHIPPED = ROOT / "config" / "dataset_profiles.json"
FILES = ["FLT_0001.CSV", "FLT_0002.CSV"]


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("UAV_PROFILES_FILE", str(SHIPPED))
    schema_llm.clear_cache()
    inf.clear_caches()
    yield
    inf.clear_caches()
    schema_llm.clear_cache()


def raw(name):
    return (DATA / name).read_bytes()


def df(name):
    return pd.read_csv(DATA / name)


def csv(d) -> bytes:
    b = io.BytesIO()
    d.to_csv(b, index=False)
    return b.getvalue()


class FakeClient:
    def __init__(self, content='{"proposals": []}'):
        self.content, self.calls = content, []

    def create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))], usage=None)


# 1 + 2 ---- recognition ---------------------------------------------------------------------------

def test_both_example_files_match_the_same_profile_and_fingerprint():
    ms = [match_profile_detailed(list(df(f).columns)) for f in FILES]
    assert all(m.kind == "exact" and m.profile.id == "uav_flt_csv" for m in ms)
    assert ms[0].fingerprint == ms[1].fingerprint == ms[0].profile.fingerprint
    assert ms[0].profile.version == "1.2.0"


def test_fingerprint_ignores_filename_row_count_order_and_case():
    h = list(df("FLT_0001.CSV").columns)
    assert headers_fingerprint(h) == headers_fingerprint(list(reversed(h))) == headers_fingerprint([x.upper() for x in h])
    short = df("FLT_0002.CSV").head(40)
    p1 = build_combined_preview(csv(short), "whatever_name.csv", {}, False)
    p2 = build_combined_preview(raw("FLT_0001.CSV"), "x.csv", {}, False)
    assert p1["assessment"]["profile_match"]["file_fingerprint"] == p2["assessment"]["profile_match"]["file_fingerprint"]


def test_future_file_with_same_schema_is_recognised_without_gpt():
    d = df("FLT_0002.CSV").sample(frac=1.0, random_state=1).rename(columns=str.upper)  # reordered rows, upper-case
    c = FakeClient()
    p = build_combined_preview(csv(d), "FLT_9999.CSV", {}, True, llm_client=c)
    assert p["assessment"]["profile_match"]["kind"] == "exact"
    assert c.calls == []  # a recognised layout needs no model call


# 3 ---- extra / omitted optional columns ------------------------------------------------------------------

def test_extra_columns_do_not_break_recognition_and_only_they_reach_gpt():
    d = df("FLT_0001.CSV")
    d["battery_v"] = 11.1
    c = FakeClient()
    p = build_combined_preview(csv(d), "x.csv", {}, True, llm_client=c)
    m = p["assessment"]["profile_match"]
    assert m["kind"] == "compatible" and m["extra_columns"] == ["battery_v"]
    sent = json.loads(c.calls[0]["messages"][1]["content"])
    assert [x["name"] for x in sent["columns"]] == ["battery_v"]


def test_omitted_optional_columns_are_tolerated_and_reported():
    d = df("FLT_0001.CSV").drop(columns=["z_set", "tof_h", "blk"])
    m = build_combined_preview(csv(d), "x.csv", {}, False)["assessment"]["profile_match"]
    assert m["kind"] == "compatible" and set(m["missing_optional"]) == {"z_set", "tof_h", "blk"}


# 4 ---- missing required columns -------------------------------------------------------------------------------

def test_missing_required_column_is_reported_and_profile_not_applied():
    d = df("FLT_0001.CSV").drop(columns=["gz"])
    p = build_combined_preview(csv(d), "x.csv", {}, False)
    a = p["assessment"]
    assert a["profile_match"]["kind"] == "partial" and a["profile_match"]["missing_required"] == ["gz"]
    assert a["outcome"] == "incompatible" and a["format_summary"] is None
    prof = next(c for c in a["checks"] if c["id"] == "profile")
    assert prof["status"] == "fail" and "gz" in prof["text"] and "NOT applied" in prof["text"]
    assert any("Required columns" in n and "gz" in n for n in a["needed"])
    by = {c["name"]: c for c in p["columns"]}
    assert by["phi_deg"]["canonical"] is None  # no silent use of the wrong profile


def test_unrelated_file_is_not_matched():
    d = pd.DataFrame({"t_ms": [0, 20, 40], "phi_deg": [0, 1, 2], "ax": [0, 0, 0]})
    assert match_profile_detailed(list(d.columns)).kind == "none"


# 5 + 6 ---- command channels are never attitude ---------------------------------------------------------------

@pytest.mark.parametrize("col,target", [("rc_yaw", "yaw"), ("rc_roll", "des_roll"), ("rc_pitch", "des_pitch"),
                                         ("rc_roll", "roll"), ("rc_yaw", "des_yaw"), ("tof_h", "alt"), ("z_est", "alt")])
def test_command_and_vertical_channels_cannot_be_forced_into_attitude(col, target):
    p = build_combined_preview(raw("FLT_0001.CSV"), "x.csv", {col: {"canonical": target, "unit": "deg"}}, False)
    by = {c["name"]: c for c in p["columns"]}
    assert by[col]["status"] == "rejected" and by[col]["canonical"] is None
    if target not in ("alt",):  # the barometer is optional for this layout, so 'alt' is not "missing"
        assert target in {m["canonical"] for m in p["assessment"]["missing"]} or target in ("roll", "des_yaw")
    assert p["assessment"]["compatible"] is False


def test_rc_yaw_is_not_yaw_even_when_an_ai_suggests_it(monkeypatch, tmp_path):
    monkeypatch.setenv("UAV_PROFILES_FILE", str(tmp_path / "none.json"))
    d = df("FLT_0001.CSV")
    prop = {"proposals": [{"source_column": "rc_yaw", "canonical": "yaw", "unit": "deg", "confidence": "high", "rationale": "r"},
                          {"source_column": "rc_roll", "canonical": "des_roll", "unit": "deg", "confidence": "high", "rationale": "r"}]}
    p = build_combined_preview(csv(d), "x.csv", {}, True, llm_client=FakeClient(json.dumps(prop)))
    by = {c["name"]: c for c in p["columns"]}
    assert by["rc_yaw"]["canonical"] is None and by["rc_roll"]["canonical"] is None
    assert {"yaw", "des_roll", "des_pitch"} <= {m["canonical"] for m in p["assessment"]["missing"]}


def test_rc_channels_are_registered_as_commands_and_retained_as_context():
    by = {c["name"]: c for c in build_combined_preview(raw("FLT_0001.CSV"), "x.csv", {}, False)["columns"]}
    for n in ("rc_roll", "rc_pitch", "rc_thr", "rc_yaw"):
        assert by[n]["profile_role"] == "command" and by[n]["status"] == "ignored" and by[n]["canonical"] is None


# 7 ---- unknown units are never silently accepted ---------------------------------------------------------------

def test_nothing_in_the_hypothesised_profile_becomes_valid_by_itself():
    p = build_combined_preview(raw("FLT_0001.CSV"), "x.csv", {}, False)
    by = {c["name"]: c for c in p["columns"]}
    for n in ("t_ms", "phi_deg", "theta_deg", "ax", "ay", "az", "gx", "gy", "gz"):
        assert by[n]["status"] == "needs_confirmation", n
        assert by[n]["conversion"] is None  # no conversion applied to an unverified unit
    assert by["t_ms"]["unit_status"] == "profile_hypothesis" and by["t_ms"]["unit_hypothesis"] == "ms"
    assert by["t_ms"]["evidence_level"] == "supported"  # shown as supported by data, but NOT applied
    assert by["phi_deg"]["unit"] is None and by["phi_deg"]["unit_hypothesis"] == "deg"
    assert by["ax"]["evidence_level"] == "supported" and by["gz"]["evidence_level"] == "unverified"  # gz has no evidence
    assert p["assessment"]["counts"]["required_mapped"] == 0
    summary = {x["label"]: x for x in p["assessment"]["format_summary"]}
    assert summary["Timestamp"]["status"] == "pending" and summary["Measured attitude"]["status"] == "pending"
    assert summary["IMU signals"]["status"] == "pending" and "axis/sign" in summary["IMU signals"]["text"]


def test_a_unit_given_with_the_upload_cannot_clear_a_frame_question():
    assign = {"t_ms": {"canonical": ci.TIME_TARGET, "unit": "ms"}, "phi_deg": {"unit": "deg"}, "theta_deg": {"unit": "deg"},
              "ax": {"unit": "m/s^2"}, "gx": {"unit": "deg/s"}}
    p = build_combined_preview(raw("FLT_0001.CSV"), "x.csv", assign, False)
    by = {c["name"]: c for c in p["columns"]}
    assert by["t_ms"]["status"] == "recognized" and by["t_ms"]["unit_status"] == "user_confirmed"
    # units alone are not enough: the axis/sign conventions are separate facts
    for n in ("phi_deg", "theta_deg"):
        assert by[n]["status"] == "needs_confirmation" and by[n]["unresolved"] == "frame"
    for n in ("ax", "gx"):
        assert by[n]["unresolved"] == "frame"
    s = {x["label"]: x for x in p["assessment"]["format_summary"]}
    assert s["Timestamp"]["status"] == "ok" and s["Measured attitude"]["status"] == "pending"


def test_axis_conventions_are_owner_choices_not_proposals():
    p = build_combined_preview(raw("FLT_0001.CSV"), "x.csv", {}, False)
    by = {c["name"]: c for c in p["columns"]}
    for n in ("phi_deg", "theta_deg", "ax", "ay", "gx", "gy", "gz"):
        assert by[n]["unresolved"] == "frame" and by[n]["frame_sign"] == 1, n  # nothing applied
    az = by["az"]
    assert az["unresolved"] == "frame" and az["evidence_level"] == "supported" and "-9.85" in az["evidence"]
    assert "CAUTION (circularity)" in by["ay"]["evidence"] and "tilt test" in by["ay"]["evidence"]


# 8 ---- timestamps ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", FILES)
def test_timestamp_normalisation_is_deterministic_and_validated(name):
    insp = ci.inspect_combined(raw(name), name)
    tm = ci.propose_mapping(insp, assignments={"t_ms": {"canonical": ci.TIME_TARGET, "unit": "ms"}},
                            profile=load_profiles()[0]).time
    assert tm.validation_status == ci.VALID
    a, b = ci.time_microseconds(insp, tm), ci.time_microseconds(insp, tm)
    assert (a == b).all() and a.iloc[0] == df(name)["t_ms"].iloc[0] * 1000
    ts = ci.check_timestamps(insp, tm)
    assert not ts["errors"] and ts["n_duplicates"] == 0 and ts["n_nonmonotonic_steps"] == 0
    assert ts["median_dt_s"] == pytest.approx(0.02) and ts["est_hz"] == pytest.approx(50.0)
    clean1, _ = normalize_timestamps(pd.DataFrame({"TimeUS": a}), stream="IMU")
    clean2, _ = normalize_timestamps(pd.DataFrame({"TimeUS": b}), stream="IMU")
    pd.testing.assert_frame_equal(clean1, clean2)


def test_duplicate_or_reversed_timestamps_are_reported():
    d = df("FLT_0001.CSV")
    d.loc[5, "t_ms"] = d.loc[4, "t_ms"]
    ts = ci.check_timestamps(ci.inspect_combined(csv(d), "x"), ci.ColumnMapping(
        source_column="t_ms", canonical=ci.TIME_TARGET, unit="ms", validation_status=ci.VALID))
    assert ts["n_duplicates"] >= 1


# 9 ---- recognised but incompatible is never "analysed" -----------------------------------------------------------------

@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("UAV_MODEL_DIR", str(ROOT / "uav_model"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "flights"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from backend import deps

    deps.get_store.cache_clear()
    inf.clear_caches()
    yield TestClient(app)
    inf.clear_caches()
    deps.get_store.cache_clear()


@pytest.mark.parametrize("name", FILES)
def test_recognised_but_incompatible_file_is_not_stored_or_analysed(client, monkeypatch, name):
    import backend.routers.flights as fl

    monkeypatch.setattr(fl, "analyze_flight", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not run")))
    r = client.post("/api/flights/combined/auto", files={"file": (name, raw(name), "text/csv")}).json()
    assert r["stage"] == "incompatible" and r["stored"] is False and r["flight_id"] is None and "analysis" not in r
    assert r["assessment"]["profile_match"]["kind"] == "exact"
    assert client.get("/api/flights").json() == []
    ids = {m["canonical"] for m in r["assessment"]["missing"]}
    assert {"yaw", "des_roll", "des_pitch"} <= ids
    # the explanation is specific to this layout
    yaw = next(n for n in r["assessment"]["needed"] if n.startswith("Measured heading"))
    assert "rc_yaw is a pilot command" in yaw and "integrating gz is rejected" in yaw
    assert not [m for m in r["assessment"]["missing"] if m["canonical"] in ("alt", "press", "temp")]  # barometer optional


# 10 ---- three-file upload untouched -----------------------------------------------------------------------------------------

def test_three_file_upload_unchanged(client):
    fr = {"IMU": _make_imu(500), "ATT": _make_att(200), "BARO": _make_baro(200)}
    r = client.post("/api/flights/upload", files={s.lower(): (f"{s}.csv", csv(f), "text/csv") for s, f in fr.items()},
                    data={"flight_id": "t3"}).json()
    assert r["ok"] and r["source"] == "individual"
    assert client.post("/api/flights/t3/analyze", json={}).json()["ok"]


# 11 ---- a fully-documented variant of this layout reproduces separate-file results through the deployed model -------------

def documented_variant(tmp_path) -> tuple[Path, pd.DataFrame]:
    """The shipped profile with every open question answered, plus the signals the detector needs.

    This is a TEST fixture: the real files cannot be scored because yaw / desired attitude are absent.
    It has no barometer columns, which the profile declares optional.
    """
    prof = json.loads(SHIPPED.read_text())
    p = prof["profiles"][0]
    signs = {"ax": -1, "ay": -1, "az": -1}  # the (roll_sign, pitch_sign) = (+1, +1) member of the family
    for k, c in p["columns"].items():
        c["unit_status"] = "verified"
        c["frame_status"] = "verified"
        c.pop("frame_expr", None)  # documented directly: explicit signs instead of owner choices
        c["frame_sign"] = signs.get(k, 1)
    p["time"]["unit_status"] = "verified"
    p["columns"].update({
        "yaw_deg": {"canonical": "yaw", "unit": "deg", "role": "signal"},
        "phi_sp": {"canonical": "des_roll", "unit": "deg", "role": "signal"},
        "theta_sp": {"canonical": "des_pitch", "unit": "deg", "role": "signal"},
    })
    path = tmp_path / "doc_profile.json"
    path.write_text(json.dumps(prof))
    d = df("FLT_0002.CSV").copy()
    n = len(d)
    d["yaw_deg"] = np.linspace(10, 40, n)
    d["phi_sp"] = d["phi_deg"].rolling(5, min_periods=1).mean()
    d["theta_sp"] = d["theta_deg"].rolling(5, min_periods=1).mean()
    return path, d


def test_documented_variant_equals_separate_files_on_the_deployed_model(tmp_path, monkeypatch):
    path, d = documented_variant(tmp_path)
    monkeypatch.setenv("UAV_PROFILES_FILE", str(path))
    res, insp, m = ci.load_combined(csv(d), "FLT_x.CSV")
    assert res.ok, res.errors
    assert m.profile_id == "uav_flt_csv" and not m.missing_required
    assert res.baro_absent and "BARO" not in res.frames and set(res.frames) == {"IMU", "ATT"}

    # Reference: the ordinary three-file flow with the polarity / unit conversions applied by hand and
    # an arbitrary barometer at the same timestamps (barometer values do not reach the model).
    t = d["t_ms"] * 1000
    sep = load_individual_csvs({
        "IMU": csv(pd.DataFrame({"TimeUS": t, "GyrX": d.gx * math.pi / 180, "GyrY": d.gy * math.pi / 180,
                                 "GyrZ": d.gz * math.pi / 180, "AccX": -d.ax, "AccY": -d.ay, "AccZ": -d.az})),
        "ATT": csv(pd.DataFrame({"TimeUS": t, "Roll": d.phi_deg, "Pitch": d.theta_deg, "Yaw": d.yaw_deg,
                                 "DesRoll": d.phi_sp, "DesPitch": d.theta_sp})),
        "BARO": csv(pd.DataFrame({"TimeUS": t, "Alt": 1.0, "Press": 101325.0, "Temp": 20.0})),
    })
    assert sep.ok, sep.errors
    # the polarity transform: az (+9.8 in the file) reaches the model as about -9.8, like its training data
    assert res.frames["IMU"]["AccZ"].median() == pytest.approx(-d["az"].median())
    assert -10.5 < res.frames["IMU"]["AccZ"].median() < -9.0

    def run(fr, meta=None):
        inf.clear_caches()
        return inf.analyze_flight(fr, flight_id="x", model_dir=str(ROOT / "uav_model"), flight_meta=meta)

    a, b = run(sep.frames), run(res.frames, {"baro_absent": True})
    assert a.ok and b.ok and a.n_windows == b.n_windows > 0
    pd.testing.assert_frame_equal(a.aligned[["_t_s", "_session_id", *a.features]], b.aligned[["_t_s", "_session_id", *b.features]],
                                  check_exact=False, atol=1e-6, rtol=1e-6)
    wa, wb = a.windows.drop(columns=["flight_id"]), b.windows.drop(columns=["flight_id"])
    pd.testing.assert_frame_equal(wa, wb, check_exact=False, atol=1e-4, rtol=1e-4)
    assert (wa.is_anomaly.to_numpy() == wb.is_anomaly.to_numpy()).all()
    assert a.threshold == b.threshold == 4.034964084625244 and a.features == b.features
    assert any("No barometer data" in w for w in b.warnings)


def test_documented_variant_end_to_end_keeps_context_untouched(client, tmp_path, monkeypatch):
    path, d = documented_variant(tmp_path)
    monkeypatch.setenv("UAV_PROFILES_FILE", str(path))
    r = client.post("/api/flights/combined/auto", files={"file": ("f.csv", csv(d), "text/csv")}).json()
    assert r["stage"] == "analyzed" and r["assessment"]["compatible"]
    s = {x["label"]: x for x in r["assessment"]["format_summary"]}
    assert s["Timestamp"]["status"] == s["Measured attitude"]["status"] == s["IMU signals"]["status"] == "ok"
    fid = r["flight_id"]
    ctx = client.get(f"/api/flights/{fid}/context", params={"limit": 2000}).json()
    assert ctx["total_rows"] == len(d) and "rc_yaw" in ctx["columns"]
    assert [row["rc_yaw"] for row in ctx["rows"]] == d["rc_yaw"].tolist()  # raw, untouched
    assert ctx["info"]["rc_yaw"]["role"] == "command"
    ov = client.get(f"/api/flights/{fid}/overview").json()["input"]
    assert ov["profile_id"] == "uav_flt_csv" and any(c["source"] == "rc_yaw" and c["ignored"] for c in ov["columns"])
    meta = client.get(f"/api/flights/{fid}").json()
    assert meta["status"] == "analyzed"


# 12 ---- profile is configuration, validated ------------------------------------------------------------------------------------

def test_shipped_profile_records_evidence_levels_and_open_questions():
    p = load_profiles()[0]
    assert p.version == "1.2.0" and p.time_unit_status == "supported" and p.time_evidence
    assert "NOT used as evidence" in p.time_evidence
    for k in ("phi_deg", "theta_deg", "ax", "ay", "az", "gx", "gy"):
        assert p.columns[k].unit_status == "supported", k
    assert p.columns["gz"].unit_status == "unverified" and p.columns["gz"].fact == "gyro_z_unit"  # own fact: no evidence
    assert set(p.choices) == {"roll_sign", "pitch_sign", "az_polarity", "gyro_z_sign"}
    exprs = {k: p.columns[k].frame_expr for k in ("phi_deg", "theta_deg", "ax", "ay", "az", "gx", "gy", "gz")}
    assert exprs == {"phi_deg": "roll_sign", "theta_deg": "pitch_sign", "ax": "-pitch_sign", "ay": "-roll_sign",
                     "az": "az_polarity", "gx": "roll_sign", "gy": "pitch_sign", "gz": "gyro_z_sign"}
    assert p.columns["rc_yaw"].never_maps_to and p.columns["tof_h"].role == "distance"
    assert "yaw" in p.missing_notes and "des_roll" in p.missing_notes and not p.baro_required
    assert len(p.columns) + 1 == 33


def test_deployed_detector_is_untouched_by_profile_work():
    import hashlib

    root = ROOT / "uav_model"
    pinned = {"lstm_autoencoder.pth": "1de2bfbaf90ca3239dd1001dd0db5cc68bd4ecbef0868acea4a07b467f396af7",
              "scaler.json": "8529145c98a1948c3700d506124454d711aa557a68b2f218bcfc18d929a18a3e",
              "model_config.json": "25755cbd676cf7f5e304361a592e647fe74a22769043c69dafd6cdfbcf693869",
              "threshold_config.json": "864318c3bf99b495dece18751b4107864de023b80fb4ef9c53d5ee31d24bb840"}
    for n, h in pinned.items():
        assert hashlib.sha256((root / n).read_bytes()).hexdigest() == h
    from src.artifact_loader import load_artifacts

    a = load_artifacts(root)
    assert a.threshold == 4.034964084625244 and a.features[0] == "gyro_x" and a.features[-1] == "pitch_res" and len(a.features) == 12
