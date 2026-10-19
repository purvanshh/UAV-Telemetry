"""Combined-CSV ingestion: mapping, validation, equivalence with the three-file flow, LLM safety."""

from __future__ import annotations

import io
import json
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
from src.upload import load_individual_csvs
from tests.conftest import _make_att, _make_baro, _make_imu

TIME_MS = {"t_ms": {"canonical": ci.TIME_TARGET, "unit": "ms"}}


def _csv(df: pd.DataFrame) -> bytes:
    b = io.BytesIO()
    df.to_csv(b, index=False)
    return b.getvalue()


def combined_from(frames: dict, time_name="TimeUS", time_scale=1.0) -> pd.DataFrame:
    """Outer-join three streams on their timestamp: one table, NaN where a group has no sample."""
    out = None
    for s in ("IMU", "ATT", "BARO"):
        f = frames[s].copy()
        out = f if out is None else out.merge(f, on="TimeUS", how="outer")
    out = out.sort_values("TimeUS").reset_index(drop=True)
    out.insert(0, time_name, out.pop("TimeUS") * time_scale)
    return out


@pytest.fixture()
def frames():
    return {"IMU": _make_imu(500), "ATT": _make_att(200), "BARO": _make_baro(200)}


@pytest.fixture(autouse=True)
def _clean(fixture_model_dir, monkeypatch, tmp_path):
    # Generic behaviour is tested with no dataset profile; profiles have their own test file.
    monkeypatch.setenv("UAV_PROFILES_FILE", str(tmp_path / "no_profiles.json"))
    schema_llm.clear_cache()
    inf.clear_caches()
    yield
    inf.clear_caches()
    schema_llm.clear_cache()


# ---- known combined file -------------------------------------------------------------------

def test_known_combined_maps_deterministically(frames):
    raw = _csv(combined_from(frames))
    res, insp, mapping = ci.load_combined(raw, "flight.csv")
    assert res.ok, res.errors
    assert mapping.scoreable and not mapping.missing_required
    assert all(c.method == "deterministic_alias" for c in mapping.columns if c.canonical)
    assert mapping.time.unit == "us" and mapping.time.unit_status == ci.UNIT_DATASET
    assert set(res.frames) == {"IMU", "ATT", "BARO"}
    assert res.source == "combined"
    assert mapping.to_dict()["version"] == ci.MAPPING_VERSION


def test_canonical_feature_names_are_documented_aliases(frames):
    df = combined_from(frames).rename(columns={"GyrX": "gyro_x", "Roll": "roll", "Alt": "alt"})
    _, _, mapping = ci.load_combined(_csv(df), "x.csv")
    assert mapping.scoreable
    gx = next(c for c in mapping.columns if c.source_column == "gyro_x")
    assert gx.canonical == "gyro_x" and gx.unit == "rad/s"


# ---- equivalence: separate files vs combined file -------------------------------------------

def _analyze(fr, model_dir):
    inf.clear_caches()
    return inf.analyze_flight(fr, flight_id="eq", model_dir=str(model_dir))


@pytest.mark.parametrize("time_name,scale,assign", [("TimeUS", 1.0, {}), ("t_ms", 1e-3, TIME_MS)])
def test_combined_equals_separate_end_to_end(frames, fixture_model_dir, time_name, scale, assign):
    sep = load_individual_csvs({s: _csv(f) for s, f in frames.items()})
    assert sep.ok
    comb, _i, m = ci.load_combined(_csv(combined_from(frames, time_name, scale)), "c.csv", assignments=assign)
    assert comb.ok, comb.errors

    # Canonical per-stream telemetry is identical for the columns the pipeline uses.
    for s in ("IMU", "ATT", "BARO"):
        cols = [c for c in comb.frames[s].columns]
        a = sep.frames[s][cols].reset_index(drop=True).astype(float)
        b = comb.frames[s][cols].reset_index(drop=True).astype(float)
        pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-9)

    ra, rb = _analyze(sep.frames, fixture_model_dir), _analyze(comb.frames, fixture_model_dir)
    assert ra.ok and rb.ok
    pd.testing.assert_frame_equal(ra.aligned, rb.aligned, check_exact=False, atol=1e-9)
    wa, wb = ra.windows.drop(columns=["flight_id"]), rb.windows.drop(columns=["flight_id"])
    pd.testing.assert_frame_equal(wa, wb, check_exact=False, atol=1e-6)
    assert (wa["is_anomaly"].to_numpy() == wb["is_anomaly"].to_numpy()).all()
    assert ra.threshold == rb.threshold and ra.features == rb.features
    assert [e.to_dict() for e in ra.events] == [e.to_dict() for e in rb.events]


# ---- the user's t_ms / phi_deg format: previewable, not scoreable --------------------------------

def user_format(n=400) -> bytes:
    rng = np.random.default_rng(3)
    t = np.arange(n) * 20
    df = pd.DataFrame(
        {
            "t_ms": t, "armed": 1, "rc_fs": 0, "tof_ok": 1,
            "rc_roll": rng.integers(1000, 2000, n), "rc_pitch": rng.integers(1000, 2000, n), "rc_thr": 1500,
            "phi_deg": rng.normal(0, 2, n), "theta_deg": rng.normal(0, 2, n), "z_est": np.linspace(0, 1, n),
            "z_set": 1.0, "u1_core": rng.normal(0, 1, n), "u1_cmd": rng.normal(0, 1, n),
            "ax": rng.normal(0, 1, n), "ay": rng.normal(0, 1, n), "az": rng.normal(-9.8, 1, n),
            "gx": rng.normal(0, 0.1, n), "gy": rng.normal(0, 0.1, n), "gz": rng.normal(0, 0.1, n),
        }
    )
    return _csv(df)


def test_user_format_previews_but_is_not_scoreable():
    p = build_combined_preview(user_format(), "user.csv", {}, llm_consent=False)
    assert p["ok"] and p["scoreable"] is False
    by = {c["name"]: c for c in p["columns"]}
    # t_ms: unit comes from the name only -> must be confirmed
    assert by["t_ms"]["role"] == "time" and by["t_ms"]["status"] == "needs_confirmation"
    assert by["t_ms"]["unit_status"] == ci.UNIT_HINT
    # nothing is silently mapped by look-alike names
    for name in ("rc_roll", "rc_pitch", "phi_deg", "theta_deg", "gx", "ax", "z_est"):
        assert by[name]["canonical"] is None and by[name]["status"] == "unknown", name
    assert "radio" in " ".join(by["rc_roll"]["notes"]).lower()
    assert by["phi_deg"]["unit_status"] == ci.UNIT_HINT and by["phi_deg"]["unit"] == "deg"
    missing = set(p["mapping"]["missing_required"])
    assert {"yaw", "des_roll", "des_pitch", "press", "temp", "gyro_x", "roll"} <= missing
    assert any("deployed detector needs" in b for b in p["blockers"])
    assert "cannot be analysed" in p["scoreable_note"]


def test_user_format_confirmed_partially_stays_blocked_no_yaw_invented():
    assign = {
        **TIME_MS,
        "phi_deg": {"canonical": "roll", "unit": "deg"}, "theta_deg": {"canonical": "pitch", "unit": "deg"},
        "gx": {"canonical": "gyro_x", "unit": "rad/s"}, "gy": {"canonical": "gyro_y", "unit": "rad/s"},
        "gz": {"canonical": "gyro_z", "unit": "rad/s"}, "ax": {"canonical": "acc_x", "unit": "m/s^2"},
        "ay": {"canonical": "acc_y", "unit": "m/s^2"}, "az": {"canonical": "acc_z", "unit": "m/s^2"},
    }
    p = build_combined_preview(user_format(), "user.csv", assign, llm_consent=False)
    assert not p["scoreable"]
    assert {"yaw", "des_roll", "des_pitch", "alt", "press", "temp"} <= set(p["mapping"]["missing_required"])
    res, _i, _m = ci.load_combined(user_format(), "user.csv", assignments=assign)
    assert not res.ok and not res.frames


def test_rc_inputs_can_never_become_desired_attitude():
    assign = {**TIME_MS, "rc_roll": {"canonical": "des_roll", "unit": "deg"},
              "rc_pitch": {"canonical": "des_pitch", "unit": "deg"}}
    p = build_combined_preview(user_format(), "u.csv", assign, llm_consent=False)
    by = {c["name"]: c for c in p["columns"]}
    for n in ("rc_roll", "rc_pitch"):
        assert by[n]["status"] == "rejected" and by[n]["canonical"] is None
    assert {"des_roll", "des_pitch"} <= set(p["mapping"]["missing_required"])


# ---- units ------------------------------------------------------------------------------------

def _full_alias_file(frames, rename, scale=None):
    df = combined_from(frames)
    for col, f in (scale or {}).items():
        df[col] = df[col] * f
    return df.rename(columns=rename)


def test_confirmed_unit_converts_exactly_and_is_recorded(frames):
    # gyro stored in deg/s under a non-dataset name; user confirms deg/s
    df = combined_from(frames)
    df["gx_dps"] = df["GyrX"] * 180 / np.pi
    df = df.drop(columns=["GyrX"])
    res, _i, m = ci.load_combined(_csv(df), "u.csv", assignments={"gx_dps": {"canonical": "gyro_x", "unit": "deg/s"}})
    assert res.ok, res.errors
    c = next(x for x in m.columns if x.source_column == "gx_dps")
    assert c.unit_status == ci.UNIT_USER and c.method == "user_confirmed" and c.scale == pytest.approx(np.pi / 180)
    got = res.frames["IMU"]["GyrX"].to_numpy()
    want = frames["IMU"]["GyrX"].to_numpy()
    np.testing.assert_allclose(got[np.isfinite(got)], want, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("unit", ["furlong", "", None, "deg"])  # deg is not a rad/s unit
def test_unsupported_or_wrong_dimension_unit_rejected(frames, unit):
    df = combined_from(frames).rename(columns={"GyrX": "gx"})
    res, _i, m = ci.load_combined(_csv(df), "u.csv", assignments={"gx": {"canonical": "gyro_x", "unit": unit}})
    assert not res.ok
    c = next(x for x in m.columns if x.source_column == "gx")
    assert c.validation_status in (ci.REJECTED, ci.NEEDS_CONFIRMATION)
    assert c.unit_status != ci.UNIT_USER and "gyro_x" in m.missing_required


def test_header_unit_hint_is_not_trusted(frames):
    df = combined_from(frames).rename(columns={"Roll": "phi_deg"})
    _, _i, m = ci.load_combined(_csv(df), "u.csv")
    c = next(x for x in m.columns if x.source_column == "phi_deg")
    assert c.canonical is None and c.unit_status == ci.UNIT_HINT
    assert "roll" in m.missing_required


# ---- timestamps (shared timebase checks) -----------------------------------------------------------

def test_wrong_time_unit_is_caught_by_plausibility(frames):
    df = combined_from(frames, "t_ms", 1e-3)  # genuinely ms
    res, _i, _m = ci.load_combined(_csv(df), "c.csv", assignments={"t_ms": {"canonical": ci.TIME_TARGET, "unit": "ns"}})
    assert not res.ok and res.errors


def test_missing_time_unit_confirmation_blocks(frames):
    res, _i, m = ci.load_combined(_csv(combined_from(frames, "t_ms", 1e-3)), "c.csv")
    assert not res.ok and m.time.validation_status == ci.NEEDS_CONFIRMATION
    assert any("unit" in e for e in res.errors)


def test_duplicate_and_nonmonotonic_timestamps_follow_existing_rules(frames):
    df = combined_from(frames)
    df.loc[10, "TimeUS"] = df.loc[8, "TimeUS"]  # duplicate within IMU rows
    res, _i, _m = ci.load_combined(_csv(df), "c.csv")
    assert any("duplicate" in w.lower() for w in res.all_warnings) or not res.ok
    df2 = combined_from(frames)
    df2["TimeUS"] = df2["TimeUS"].to_numpy()[::-1]  # fully reversed
    res2, _i, _m = ci.load_combined(_csv(df2), "c.csv")
    assert not res2.ok or any("order" in w.lower() or "monoton" in w.lower() for w in res2.all_warnings)


# ---- file-level problems ---------------------------------------------------------------------------

def test_duplicate_headers_empty_and_header_only_rejected():
    for raw, needle in [
        (b"TimeUS,GyrX,GyrX\n1,2,3\n", "Duplicate"),
        (b"TimeUS,gyrx,Gyr_X\n1,2,3\n", "Duplicate"),
        (b"", "empty"),
        (b"TimeUS,GyrX\n", "no data rows"),
    ]:
        res, insp, _m = ci.load_combined(raw, "x.csv")
        assert not res.ok and needle in res.errors[0], (raw, res.errors)


def test_oversized_combined_rejected(monkeypatch, frames):
    monkeypatch.setenv("UAV_MAX_UPLOAD_MB", "0.001")
    res, _i, _m = ci.load_combined(_csv(combined_from(frames)), "big.csv")
    assert not res.ok and "upload limit" in res.errors[0]


def test_non_numeric_required_column_rejected(frames):
    df = combined_from(frames).astype({"GyrX": object})
    df.loc[df.index[::2], "GyrX"] = "bad"
    res, _i, m = ci.load_combined(_csv(df), "c.csv")
    assert not res.ok
    assert next(c for c in m.columns if c.source_column == "GyrX").validation_status == ci.REJECTED


def test_few_bad_cells_become_missing_with_warning(frames):
    df = combined_from(frames).astype({"AccX": object})
    idx = df["AccX"].first_valid_index()
    df.loc[idx, "AccX"] = "oops"
    res, _i, _m = ci.load_combined(_csv(df), "c.csv")
    assert res.ok and any("AccX" in w and "non-numeric" in w for w in res.all_warnings)
    assert np.isfinite(res.frames["IMU"]["AccX"]).sum() == frames["IMU"]["AccX"].notna().sum() - 1


def test_infinite_values_and_missing_values_are_not_passed_downstream(frames):
    df = combined_from(frames)
    i = df["GyrY"].first_valid_index()
    df.loc[i, "GyrY"] = np.inf
    res, _i, _m = ci.load_combined(_csv(df).replace(b"inf", b"inf"), "c.csv")
    assert res.ok
    assert not np.isinf(res.frames["IMU"]["GyrY"].to_numpy()).any()


def test_two_columns_claiming_same_signal_are_ambiguous(frames):
    df = combined_from(frames)
    df["gyro_x"] = df["GyrX"]  # canonical alias AND dataset name present
    res, _i, m = ci.load_combined(_csv(df), "c.csv")
    assert not res.ok
    assert sum(c.validation_status == ci.AMBIGUOUS for c in m.columns) == 2


# ---- LLM proposer: fallback and containment -------------------------------------------------------

class FakeClient:
    def __init__(self, content=None, exc=None):
        self.content, self.exc, self.calls = content, exc, []

    def create(self, **kw):
        self.calls.append(kw)
        if self.exc:
            raise self.exc
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )


def _proposal(src, canon, unit="rad/s", conf="high"):
    return {"source_column": src, "canonical": canon, "unit": unit, "confidence": conf, "rationale": "r"}


def test_no_consent_means_no_external_call():
    c = FakeClient(content="{}")
    p = build_combined_preview(user_format(), "u.csv", {}, llm_consent=False, llm_client=c)
    assert c.calls == [] and p["llm"]["status"] == "not_requested"


def test_missing_api_key_falls_back(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    p = build_combined_preview(user_format(), "u.csv", {}, llm_consent=True)
    assert p["llm"]["status"] == "skipped_no_key" and p["scoreable"] is False


def test_valid_proposals_are_only_suggestions():
    c = FakeClient(content=json.dumps({"proposals": [_proposal("gx", "gyro_x", "rad/s")]}))
    p = build_combined_preview(user_format(), "u.csv", TIME_MS, llm_consent=True, llm_client=c)
    gx = next(x for x in p["columns"] if x["name"] == "gx")
    assert gx["status"] == "needs_confirmation" and gx["method"] == "llm_proposed"
    assert gx["unit_status"] == ci.UNIT_LLM and p["scoreable"] is False
    # request carried only names, <=5 samples and summaries, plus the schema; strict schema requested
    req = c.calls[0]
    assert req["response_format"]["json_schema"]["strict"] is True and req["max_tokens"] <= 1500
    user = json.loads(req["messages"][1]["content"])
    assert all(len(col["samples"]) <= 5 for col in user["columns"])
    assert "rows" not in user and len(json.dumps(user)) <= schema_llm.MAX_REQUEST_CHARS


def test_llm_cannot_map_rc_to_desired_or_invent_targets():
    c = FakeClient(content=json.dumps({"proposals": [
        _proposal("rc_roll", "des_roll", "deg"), _proposal("z_est", "made_up"), _proposal("ghost", "roll"),
        _proposal("gx", "gyro_x"), _proposal("gy", "gyro_x"),
    ]}))
    p = build_combined_preview(user_format(), "u.csv", TIME_MS, llm_consent=True, llm_client=c)
    by = {x["name"]: x for x in p["columns"]}
    assert by["rc_roll"]["canonical"] is None and by["z_est"]["canonical"] is None
    assert by["gx"]["canonical"] == "gyro_x" and by["gy"]["canonical"] is None  # duplicate target dropped
    assert len(p["llm"]["dropped"]) >= 3


@pytest.mark.parametrize("content", ["not json", "[]", '{"proposals": "x"}', '{"proposals": [{"source_column": "gx"}]}',
                                     '{"proposals": [], "extra": 1}'])
def test_invalid_structured_output_is_ignored(content):
    c = FakeClient(content=content)
    p = build_combined_preview(user_format(), "u.csv", {}, llm_consent=True, llm_client=c)
    assert p["llm"]["status"] == "invalid_output" and p["ok"] and not p["scoreable"]
    assert all(x["method"] != "llm_proposed" for x in p["columns"])


def test_timeout_and_api_errors_do_not_break_preview(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    for exc in (TimeoutError("timed out"), RuntimeError("boom")):
        c = FakeClient(exc=exc)
        p = build_combined_preview(user_format(), "u.csv", {}, llm_consent=True, llm_client=c)
        assert p["llm"]["status"] == "error" and p["ok"]
        assert "boom" not in json.dumps(p["llm"])  # raw exception text is not echoed


def test_identical_schema_reuses_cached_proposals():
    c = FakeClient(content=json.dumps({"proposals": [_proposal("gx", "gyro_x")]}))
    build_combined_preview(user_format(), "u.csv", {}, llm_consent=True, llm_client=c)
    p2 = build_combined_preview(user_format(), "u2.csv", {}, llm_consent=True, llm_client=c)
    assert len(c.calls) == 1 and p2["llm"]["status"] == "cached"


def test_llm_output_never_makes_a_mapping_valid_and_never_touches_detector(frames):
    from src.artifact_loader import load_artifacts
    from pathlib import Path

    before = load_artifacts(Path(__file__).resolve().parents[1] / "uav_model").threshold
    df = combined_from(frames).rename(columns={"GyrX": "gx"})
    c = FakeClient(content=json.dumps({"proposals": [_proposal("gx", "gyro_x")]}))
    p = build_combined_preview(_csv(df), "x.csv", {}, llm_consent=True, llm_client=c)
    assert p["scoreable"] is False  # gx still awaits confirmation
    assert load_artifacts(Path(__file__).resolve().parents[1] / "uav_model").threshold == before == 4.034964084625244


# ---- API ---------------------------------------------------------------------------------------------

@pytest.fixture()
def client(tmp_path, monkeypatch, fixture_model_dir):
    monkeypatch.setenv("UAV_MODEL_DIR", str(fixture_model_dir))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "flights"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from backend import deps

    deps.get_store.cache_clear()
    inf.clear_caches()
    yield TestClient(app)
    inf.clear_caches()
    deps.get_store.cache_clear()


def test_api_preview_and_upload_scoreable_combined(client, frames):
    raw = _csv(combined_from(frames))
    pv = client.post("/api/flights/combined/preview", files={"file": ("f.csv", raw, "text/csv")}).json()
    assert pv["ok"] and pv["scoreable"] and pv["counts"]["recognized"] >= 15
    up = client.post("/api/flights/combined/upload", files={"file": ("f.csv", raw, "text/csv")},
                     data={"flight_id": "combo-1"}).json()
    assert up["ok"] and up["source"] == "combined"
    meta = client.get("/api/flights/combo-1").json()
    assert meta["status"] == "validated"
    a = client.post("/api/flights/combo-1/analyze", json={}).json()
    assert a["ok"] and a["n_windows"] > 0
    # provenance: mapping version + file hash stored, no secrets
    from backend.deps import get_store

    m = get_store().load_meta("combo-1")["combined"]
    assert m["mapping_version"] == ci.MAPPING_VERSION and len(m["sha256"]) == 64
    assert "api_key" not in json.dumps(m).lower()


def test_api_upload_of_unscoreable_file_stores_nothing(client):
    r = client.post("/api/flights/combined/upload", files={"file": ("u.csv", user_format(), "text/csv")},
                    data={"assignments": json.dumps(TIME_MS)}).json()
    assert r["ok"] is False and r["errors"]
    assert client.get("/api/flights").json() == []


def test_api_bad_assignments_json_is_400(client):
    r = client.post("/api/flights/combined/preview", files={"file": ("u.csv", user_format(), "text/csv")},
                    data={"assignments": "{not json"})
    assert r.status_code == 400


def test_existing_three_file_upload_still_works(client, frames):
    r = client.post("/api/flights/upload", files={s.lower(): (f"{s}.csv", _csv(f), "text/csv") for s, f in frames.items()},
                    data={"flight_id": "legacy-3"}).json()
    assert r["ok"] and r["source"] == "individual"
