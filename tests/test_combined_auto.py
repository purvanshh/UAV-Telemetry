"""Automatic single-CSV workflow: schema detection, unit conversion, gating, GPT containment.

Equivalence and invariance tests run against the real deployed artifacts in ``uav_model/``.
"""

from __future__ import annotations

import hashlib
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
from src.dataset_profiles import ProfileError, load_profiles
from src.upload import load_individual_csvs
from tests.conftest import _make_att, _make_baro, _make_imu

ROOT = Path(__file__).resolve().parents[1]
DEPLOYED = ROOT / "uav_model"

# A fully documented synthetic layout: every unit is stated, several need exact conversion.
PROFILE = {
    "version": 1,
    "profiles": [{
        "id": "test_documented_v1",
        "description": "synthetic",
        "documentation_status": "test",
        "signature": ["t_ms", "phi_deg", "theta_deg", "yaw_deg", "gx", "ax"],
        "time": {"column": "t_ms", "unit": "ms"},
        "columns": {
            "phi_deg": {"canonical": "roll", "unit": "deg"}, "theta_deg": {"canonical": "pitch", "unit": "deg"},
            "yaw_deg": {"canonical": "yaw", "unit": "deg"},
            "phi_sp": {"canonical": "des_roll", "unit": "deg"}, "theta_sp": {"canonical": "des_pitch", "unit": "deg"},
            "gx": {"canonical": "gyro_x", "unit": "deg/s"}, "gy": {"canonical": "gyro_y", "unit": "deg/s"},
            "gz": {"canonical": "gyro_z", "unit": "deg/s"},
            "ax": {"canonical": "acc_x", "unit": "g"}, "ay": {"canonical": "acc_y", "unit": "g"},
            "az": {"canonical": "acc_z", "unit": "g"},
            "alt_cm": {"canonical": "alt", "unit": "cm"}, "p_hpa": {"canonical": "press", "unit": "hPa"},
            "t_c": {"canonical": "temp", "unit": "degC"},
        },
        "ignored": ["armed", "rc_roll", "rc_pitch"],
        "ignored_patterns": ["^m[0-9]+$"],
    }],
}


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    p = tmp_path / "profiles.json"
    p.write_text(json.dumps(PROFILE))
    monkeypatch.setenv("UAV_PROFILES_FILE", str(p))
    schema_llm.clear_cache()
    inf.clear_caches()
    yield
    inf.clear_caches()
    schema_llm.clear_cache()


def _csv(df) -> bytes:
    b = io.BytesIO()
    df.to_csv(b, index=False)
    return b.getvalue()


@pytest.fixture()
def frames():
    return {"IMU": _make_imu(500), "ATT": _make_att(200), "BARO": _make_baro(200)}


def documented_file(frames, extra=True) -> pd.DataFrame:
    """The same flight, expressed in the documented layout and its non-canonical units."""
    out = None
    for s in ("IMU", "ATT", "BARO"):
        out = frames[s] if out is None else out.merge(frames[s], on="TimeUS", how="outer")
    out = out.sort_values("TimeUS").reset_index(drop=True)
    d = pd.DataFrame({"t_ms": out["TimeUS"] / 1000.0})
    d["phi_deg"], d["theta_deg"], d["yaw_deg"] = out["Roll"], out["Pitch"], out["Yaw"]
    d["phi_sp"], d["theta_sp"] = out["DesRoll"], out["DesPitch"]
    d["gx"], d["gy"], d["gz"] = (out[c] * 180 / math.pi for c in ("GyrX", "GyrY", "GyrZ"))
    d["ax"], d["ay"], d["az"] = (out[c] / 9.80665 for c in ("AccX", "AccY", "AccZ"))
    d["alt_cm"], d["p_hpa"], d["t_c"] = out["Alt"] * 100, out["Press"] / 100, out["Temp"]
    if extra:
        d["armed"], d["rc_roll"], d["rc_pitch"], d["m1"], d["m2"] = 1, 1500, 1500, 0.1, 0.2
        d["mystery_sensor"] = 3.3
        d["status_text"] = "ok"
    return d


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("UAV_MODEL_DIR", str(DEPLOYED))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "flights"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from backend import deps

    deps.get_store.cache_clear()
    inf.clear_caches()
    yield TestClient(app)
    inf.clear_caches()
    deps.get_store.cache_clear()


def auto(client, raw, **data):
    return client.post("/api/flights/combined/auto", files={"file": ("f.csv", raw, "text/csv")}, data=data).json()


# 1 ---- compatible files need no confirmation --------------------------------------------------

def test_documented_file_is_mapped_converted_and_analysed_automatically(client, frames):
    r = auto(client, _csv(documented_file(frames)))
    assert r["stage"] == "analyzed" and r["stored"] and r["flight_id"]
    a = r["assessment"]
    assert a["outcome"] == "ready" and a["compatible"]
    assert a["counts"]["required_mapped"] == a["counts"]["required_total"]
    assert a["counts"]["ignored"] >= 4 and a["counts"]["unrecognised"] == 2  # mystery_sensor, status_text
    assert all(c["status"] in ("ok",) for c in a["checks"])
    assert all(c["method"] in ("dataset_profile", "deterministic_alias") for c in r["columns"] if c["role"] == "signal")
    assert r["analysis"]["ok"] and r["analysis"]["n_windows"] > 0


def test_dataset_named_file_needs_no_profile_or_confirmation(client, frames, monkeypatch, tmp_path):
    monkeypatch.setenv("UAV_PROFILES_FILE", str(tmp_path / "none.json"))
    out = None
    for s in ("IMU", "ATT", "BARO"):
        out = frames[s] if out is None else out.merge(frames[s], on="TimeUS", how="outer")
    r = auto(client, _csv(out.sort_values("TimeUS")))
    assert r["stage"] == "analyzed"


# 2 ---- unit conversions -------------------------------------------------------------------------

def test_supported_conversions_produce_canonical_values(frames):
    d = documented_file(frames)
    res, insp, m = ci.load_combined(_csv(d), "d.csv")
    assert res.ok, res.errors
    imu = res.frames["IMU"].dropna(subset=["GyrX"]).reset_index(drop=True)
    np.testing.assert_allclose(imu["GyrX"], frames["IMU"]["GyrX"], rtol=1e-9, atol=1e-12)  # deg/s -> rad/s
    np.testing.assert_allclose(imu["AccZ"], frames["IMU"]["AccZ"], rtol=1e-9)  # g -> m/s^2
    assert res.frames["BARO"]["Press"].dropna().iloc[0] == pytest.approx(101325.0)  # hPa -> Pa
    assert res.frames["BARO"]["Alt"].dropna().iloc[-1] == pytest.approx(10.0)  # cm -> m
    assert res.frames["IMU"]["TimeUS"].iloc[0] == pytest.approx(1_000_000)  # ms -> us
    byname = {c.source_column: c for c in m.columns}
    assert byname["gx"].unit_status == ci.UNIT_PROFILE and byname["gx"].scale == pytest.approx(math.pi / 180)


# 3 ---- unknown / irrelevant columns ---------------------------------------------------------------

def test_unknown_columns_do_not_break_ingestion(frames):
    d = documented_file(frames)
    d["???"] = "x"
    d["Δ weird name"] = 1.0
    res, insp, m = ci.load_combined(_csv(d), "d.csv")
    assert res.ok
    unused = {c.source_column for c in m.columns if not c.canonical}
    assert {"???", "Δ weird name", "mystery_sensor", "armed", "rc_roll", "m1"} <= unused


# 4 ---- ambiguous units are never silently accepted ----------------------------------------------------

def user_format(n=400, with_extra=True) -> bytes:
    rng = np.random.default_rng(3)
    df = pd.DataFrame({
        "t_ms": np.arange(n) * 20, "armed": 1, "rc_fs": 0, "tof_ok": 1,
        "rc_roll": rng.integers(1000, 2000, n), "rc_pitch": rng.integers(1000, 2000, n), "rc_thr": 1500,
        "phi_deg": rng.normal(0, 2, n), "theta_deg": rng.normal(0, 2, n), "z_est": np.linspace(0, 1, n),
        "z_set": 1.0, "u1_core": rng.normal(0, 1, n), "u1_cmd": rng.normal(0, 1, n),
        "ax": rng.normal(0, 1, n), "ay": rng.normal(0, 1, n), "az": rng.normal(-9.8, 1, n),
        "gx": rng.normal(0, 0.1, n), "gy": rng.normal(0, 0.1, n), "gz": rng.normal(0, 0.1, n),
    })
    return _csv(df)


def default_profile_env(monkeypatch):
    monkeypatch.setenv("UAV_PROFILES_FILE", str(ROOT / "config" / "dataset_profiles.json"))


def test_shipped_profile_accepts_documented_parts_and_leaves_units_unresolved(monkeypatch):
    default_profile_env(monkeypatch)
    p = build_combined_preview(user_format(), "u.csv", {}, llm_consent=False)
    by = {c["name"]: c for c in p["columns"]}
    assert by["t_ms"]["status"] == "recognized" and by["t_ms"]["unit_status"] == "profile"
    assert by["phi_deg"]["canonical"] == "roll" and by["phi_deg"]["status"] == "recognized"
    for g in ("gx", "gy", "gz", "ax", "ay", "az"):
        assert by[g]["status"] == "needs_confirmation" and by[g]["unresolved"] == "unit", g
        assert "not documented" in by[g]["ambiguity"]
    for ign in ("armed", "rc_roll", "rc_pitch", "rc_thr", "z_est", "u1_core"):
        assert by[ign]["status"] == "ignored" or by[ign]["canonical"] is None
    a = p["assessment"]
    assert a["outcome"] == "incompatible" and not a["compatible"]
    assert {m["canonical"] for m in a["missing"]} >= {"yaw", "des_roll", "des_pitch"}
    assert any("yaw" in n.lower() for n in a["needed"])
    groups = {g["key"]: g["status"] for g in a["detector_inputs"]}
    assert groups["attitude"] == "ok" and groups["gyro"] == "unit_unresolved" and groups["yaw"] == "missing"


def test_gpt_cannot_supply_a_unit(monkeypatch, tmp_path):
    monkeypatch.setenv("UAV_PROFILES_FILE", str(tmp_path / "none.json"))
    c = FakeClient(content=json.dumps({"proposals": [_p("gx", "gyro_x", "rad/s", "high")]}))
    p = build_combined_preview(user_format(), "u.csv", {}, llm_consent=True, llm_client=c)
    gx = next(x for x in p["columns"] if x["name"] == "gx")
    assert gx["status"] == "needs_confirmation" and gx["unresolved"] == "unit" and gx["semantics"] == "ai_validated"
    assert gx["unit_status"] == "llm_stated" and gx["conversion"] is None
    # a unit from the user completes just that field (no canonical needed)
    p2 = build_combined_preview(user_format(), "u.csv", {"gx": {"unit": "deg/s"}}, llm_consent=True, llm_client=c)
    gx2 = next(x for x in p2["columns"] if x["name"] == "gx")
    assert gx2["status"] == "recognized" and gx2["unit_status"] == "user_confirmed"
    assert gx2["conversion"]["scale"] == pytest.approx(math.pi / 180)
    assert len(c.calls) == 1  # second preview reused the cached proposal


def test_gpt_proposal_with_implausible_values_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("UAV_PROFILES_FILE", str(tmp_path / "none.json"))
    df = pd.read_csv(io.BytesIO(user_format()))
    df["gx"] = df["gx"] * 1e6
    c = FakeClient(content=json.dumps({"proposals": [_p("gx", "gyro_x", "rad/s", "high")]}))
    p = build_combined_preview(_csv(df), "u.csv", {}, llm_consent=True, llm_client=c)
    gx = next(x for x in p["columns"] if x["name"] == "gx")
    assert gx["canonical"] is None and gx["status"] == "rejected"


def test_low_confidence_proposal_leaves_meaning_unresolved(monkeypatch, tmp_path):
    monkeypatch.setenv("UAV_PROFILES_FILE", str(tmp_path / "none.json"))
    c = FakeClient(content=json.dumps({"proposals": [_p("gx", "gyro_x", None, "low")]}))
    p = build_combined_preview(user_format(), "u.csv", {}, llm_consent=True, llm_client=c)
    gx = next(x for x in p["columns"] if x["name"] == "gx")
    assert gx["unresolved"] == "meaning" and gx["semantics"] != "ai_validated"


# 5 + 12 ---- missing signals prevent inference; nothing is reported as analysed ------------------------

def test_missing_required_signals_never_reach_the_detector(client, monkeypatch):
    default_profile_env(monkeypatch)
    import backend.routers.flights as fl

    def boom(*a, **k):
        raise AssertionError("detector must not run for an incompatible file")

    monkeypatch.setattr(fl, "analyze_flight", boom)
    r = auto(client, user_format())
    assert r["stage"] == "incompatible" and r["stored"] is False and r["flight_id"] is None
    assert "analysis" not in r
    assert client.get("/api/flights").json() == []
    assert r["assessment"]["headline"] == "The deployed detector cannot analyse this file"


def test_needs_input_stage_when_only_units_are_missing(client, frames):
    d = documented_file(frames)
    prof = json.loads(Path(__import__("os").environ["UAV_PROFILES_FILE"]).read_text())
    prof["profiles"][0]["columns"]["gx"]["unit"] = None
    Path(__import__("os").environ["UAV_PROFILES_FILE"]).write_text(json.dumps(prof))
    r = auto(client, _csv(d))
    assert r["stage"] == "needs_input" and r["stored"] is False
    assert [u["column"] for u in r["assessment"]["unresolved"]] == ["gx"]
    # supplying only that unit completes the flow automatically
    r2 = auto(client, _csv(d), assignments=json.dumps({"gx": {"unit": "deg/s"}}))
    assert r2["stage"] == "analyzed"


def test_missing_yaw_is_never_invented(frames):
    d = documented_file(frames).drop(columns=["yaw_deg"])
    res, insp, m = ci.load_combined(_csv(d), "d.csv")
    assert not res.ok and "yaw" in m.missing_required and not res.frames


# 6 ---- stick commands -------------------------------------------------------------------------------

def test_rc_inputs_are_never_desired_attitude(frames):
    d = documented_file(frames).drop(columns=["phi_sp", "theta_sp"])
    d["rc_roll"], d["rc_pitch"] = 1500.0, 1500.0
    assign = {"rc_roll": {"canonical": "des_roll", "unit": "deg"}, "rc_pitch": {"canonical": "des_pitch", "unit": "deg"}}
    res, _i, m = ci.load_combined(_csv(d), "d.csv", assignments=assign)
    assert not res.ok and {"des_roll", "des_pitch"} <= set(m.missing_required)
    c = FakeClient(content=json.dumps({"proposals": [_p("rc_roll", "des_roll", "deg", "high")]}))
    p = build_combined_preview(_csv(d), "d.csv", {}, llm_consent=True, llm_client=c)
    assert p["assessment"]["outcome"] == "incompatible"


# 7 + 8 ---- GPT can't bypass validation; outage doesn't break deterministic mapping --------------------------

class FakeClient:
    def __init__(self, content=None, exc=None):
        self.content, self.exc, self.calls = content, exc, []

    def create(self, **kw):
        self.calls.append(kw)
        if self.exc:
            raise self.exc
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
                               usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))


def _p(src, canon, unit="rad/s", conf="high"):
    return {"source_column": src, "canonical": canon, "unit": unit, "confidence": conf, "rationale": "r"}


@pytest.mark.parametrize("content", ["nope", "[]", '{"proposals": 3}', '{"proposals": [{"source_column": "mystery_sensor"}]}',
                                     json.dumps({"proposals": [_p("mystery_sensor", "not_a_signal")]}),
                                     json.dumps({"proposals": [_p("ghost_col", "roll")]})])
def test_invalid_gpt_output_cannot_change_a_decision(frames, content):
    d = documented_file(frames)
    base = build_combined_preview(_csv(d), "d.csv", {}, llm_consent=False)
    got = build_combined_preview(_csv(d), "d.csv", {}, llm_consent=True, llm_client=FakeClient(content=content))
    assert got["assessment"]["outcome"] == base["assessment"]["outcome"] == "ready"
    assert [c["canonical"] for c in got["columns"]] == [c["canonical"] for c in base["columns"]]


def test_gpt_cannot_override_a_documented_mapping(frames):
    d = documented_file(frames)
    c = FakeClient(content=json.dumps({"proposals": [_p("gx", "acc_x", "g", "high"), _p("mystery_sensor", "temp", "degC")]}))
    got = build_combined_preview(_csv(d), "d.csv", {}, llm_consent=True, llm_client=c)
    by = {x["name"]: x for x in got["columns"]}
    assert by["gx"]["canonical"] == "gyro_x" and by["gx"]["method"] == "dataset_profile"
    assert by["mystery_sensor"]["canonical"] in (None, "temp")  # at most a pending suggestion
    assert by["t_c"]["canonical"] == "temp" and by["t_c"]["status"] == "recognized"
    assert got["assessment"]["compatible"]  # the decision is made by the validators, not the model


@pytest.mark.parametrize("exc", [TimeoutError("t"), RuntimeError("down")])
def test_api_outage_keeps_deterministic_flow(client, frames, monkeypatch, exc):
    monkeypatch.setattr("time.sleep", lambda s: None)
    p = build_combined_preview(_csv(documented_file(frames)), "d.csv", {}, llm_consent=True, llm_client=FakeClient(exc=exc))
    assert p["assessment"]["compatible"]
    r = auto(client, _csv(documented_file(frames)), llm_consent="true")  # no key configured in the test server
    assert r["stage"] == "analyzed" and r["llm"]["status"] in ("nothing_to_ask", "skipped_no_key")


def test_only_unfamiliar_columns_are_sent_to_gpt(frames):
    d = documented_file(frames)
    c = FakeClient(content=json.dumps({"proposals": []}))
    build_combined_preview(_csv(d), "d.csv", {}, llm_consent=True, llm_client=c)
    sent = json.loads(c.calls[0]["messages"][1]["content"])
    assert {x["name"] for x in sent["columns"]} == {"mystery_sensor", "status_text"}  # not gx, armed, rc_roll, ...


# 9 ---- three-file upload unchanged ---------------------------------------------------------------------

def test_three_file_upload_still_works(client, frames):
    r = client.post("/api/flights/upload", files={s.lower(): (f"{s}.csv", _csv(f), "text/csv") for s, f in frames.items()},
                    data={"flight_id": "legacy-3"}).json()
    assert r["ok"] and r["source"] == "individual"
    a = client.post("/api/flights/legacy-3/analyze", json={}).json()
    assert a["ok"] and a["n_windows"] > 0


# 10 ---- equivalence with the DEPLOYED checkpoint ----------------------------------------------------------

def test_single_file_equals_separate_files_with_deployed_model(frames):
    sep = load_individual_csvs({s: _csv(f) for s, f in frames.items()})
    comb, _i, _m = ci.load_combined(_csv(documented_file(frames)), "d.csv")
    assert sep.ok and comb.ok, comb.errors

    def run(fr):
        inf.clear_caches()
        return inf.analyze_flight(fr, flight_id="x", model_dir=str(DEPLOYED))

    ra, rb = run(sep.frames), run(comb.frames)
    assert ra.ok and rb.ok and ra.n_windows == rb.n_windows > 0
    pd.testing.assert_frame_equal(ra.aligned, rb.aligned, check_exact=False, atol=1e-6, rtol=1e-6)
    wa, wb = ra.windows.drop(columns=["flight_id"]), rb.windows.drop(columns=["flight_id"])
    pd.testing.assert_frame_equal(wa, wb, check_exact=False, atol=1e-4, rtol=1e-4)
    assert (wa["is_anomaly"].to_numpy() == wb["is_anomaly"].to_numpy()).all()
    assert (wa[["t_start", "t_end", "session_id"]].to_numpy() == wb[["t_start", "t_end", "session_id"]].to_numpy()).all()
    assert ra.threshold == rb.threshold == 4.034964084625244 and ra.features == rb.features
    assert [e.event_id for e in ra.events] == [e.event_id for e in rb.events]


# 11 ---- deployed artifacts unchanged ------------------------------------------------------------------------

PINNED = {
    "lstm_autoencoder.pth": "1de2bfbaf90ca3239dd1001dd0db5cc68bd4ecbef0868acea4a07b467f396af7",
    "scaler.json": "8529145c98a1948c3700d506124454d711aa557a68b2f218bcfc18d929a18a3e",
    "model_config.json": "25755cbd676cf7f5e304361a592e647fe74a22769043c69dafd6cdfbcf693869",
    "threshold_config.json": "864318c3bf99b495dece18751b4107864de023b80fb4ef9c53d5ee31d24bb840",
    "feature_schema.json": "75170a249c52de800af1675fb4eccbd037a3068d8ba1944dc2aa5505612d98da",
    "preprocessing.py": "d13b0e5d389e9abf137782d5632da07b1710e8131764c076e29da81912513249",
}


@pytest.mark.parametrize("name,digest", sorted(PINNED.items()))
def test_deployed_artifact_bytes_are_unchanged(name, digest):
    assert hashlib.sha256((DEPLOYED / name).read_bytes()).hexdigest() == digest


# ---- profile file validation ------------------------------------------------------------------------------------

def test_profile_with_unconvertible_unit_is_rejected(tmp_path):
    bad = json.loads(json.dumps(PROFILE))
    bad["profiles"][0]["columns"]["gx"]["unit"] = "furlong"
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ProfileError):
        load_profiles(p)
    bad["profiles"][0]["columns"]["gx"] = {"canonical": "nope", "unit": "rad/s"}
    p.write_text(json.dumps(bad))
    with pytest.raises(ProfileError):
        load_profiles(p)


def test_profile_requires_every_signature_header(frames):
    d = documented_file(frames).drop(columns=["gx"])  # signature header missing -> profile does not apply
    p = build_combined_preview(_csv(d), "d.csv", {}, llm_consent=False)
    by = {c["name"]: c for c in p["columns"]}
    assert by["phi_deg"]["canonical"] is None and by["t_ms"]["status"] != "recognized"
    assert not p["assessment"]["compatible"]


def test_overview_exposes_how_the_file_was_read(client, frames):
    r = auto(client, _csv(documented_file(frames)))
    ov = client.get(f"/api/flights/{r['flight_id']}/overview").json()
    inp = ov["input"]
    assert inp["profile_id"] == "test_documented_v1" and inp["mapping_version"] == ci.MAPPING_VERSION
    gx = next(c for c in inp["columns"] if c["source"] == "gx")
    assert gx["canonical"] == "gyro_x" and gx["converted"] and gx["unit"] == "deg/s"
    assert "api" not in json.dumps(inp).lower()
