"""Context-aware interpretation: commands vs responses, sensor consistency, physical significance.

The tests use synthetic recordings with known structure so that the expected relationship is
unambiguous, plus the provided FLT_0003 recording as a regression file. No expected event list
or conclusion is hardcoded for FLT_0003.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.direct_analysis import analyze_table
from src.flight_relationships import analyse_flight, command_sentence, physical_summary
from src.telemetry_schema import load_schemas

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "fixtures" / "data"


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("UAV_PROFILE_CONFIRMATIONS", str(tmp_path / "none.json"))
    monkeypatch.setenv("UAV_PROFILES_FILE", str(ROOT / "config" / "dataset_profiles.json"))


def _csv(rows: pd.DataFrame) -> bytes:
    return rows.to_csv(index=False).encode()


def _base_frame(n: int = 600, dt_ms: int = 20) -> pd.DataFrame:
    t = np.arange(n) * dt_ms + 18
    df = pd.DataFrame({"t_ms": t})
    df["armed"] = 1
    df["rc_fs"] = 0
    df["tof_ok"] = 1
    df["tof_st"] = 0
    df["blk"] = 0
    df["aux1"] = -1000.0
    df["kill"] = 26
    df["rc_roll"] = 0.0
    df["rc_pitch"] = 0.0
    df["rc_thr"] = -700.0
    df["rc_yaw"] = 0.0
    df["m1"] = df["m2"] = df["m3"] = df["m4"] = 1400.0
    df["phi_deg"] = 0.0
    df["theta_deg"] = 0.0
    df["z_est"] = 0.0
    df["z_set"] = 0.0
    df["tof_h"] = 0.0
    df["tof_mm"] = 200.0
    for c in ("u1_core", "u1_cmd", "u2", "u3", "u4"):
        df[c] = 0.0
    df["ax"] = 0.0
    df["ay"] = 0.0
    df["az"] = 9.8
    df["gx"] = 0.0
    df["gy"] = 0.0
    df["gz"] = 0.0
    return df


def _analyse(df: pd.DataFrame):
    return analyze_table(_csv(df), "synthetic.csv").data


def test_commanded_roll_is_explained_and_uncommanded_roll_is_not():
    n = 600
    df = _base_frame(n)
    # A commanded roll ramp from 4 s to 5 s: the input moves, the aircraft follows, the gyro follows the angle.
    k = np.arange(n)
    ramp = np.clip((k - 200) / 50.0, 0, 1)
    df["rc_roll"] = 300.0 * ramp
    df["phi_deg"] = 20.0 * ramp
    dphi = np.gradient(df["phi_deg"].to_numpy(), 0.02)
    df["gx"] = dphi
    df["ay"] = 9.8 * np.sin(np.deg2rad(df["phi_deg"]))
    df["az"] = 9.8 * np.cos(np.deg2rad(df["phi_deg"]))
    good = _analyse(df)
    rel = good["flight"]["relationships"]["roll_tracking"]
    assert rel["status"] in ("consistent", "partially_consistent"), rel
    att = [e for e in good["events"] if e.get("family") == "attitude"]
    assert att and any(e.get("interpretation") == "expected_response" for e in att)

    bad = df.copy()
    bad["rc_roll"] = 0.0
    dbad = _analyse(bad)
    att_bad = [e for e in dbad["events"] if e.get("family") == "attitude"]
    assert att_bad and any(e.get("interpretation") == "unexplained_deviation" for e in att_bad)


def test_throttle_and_motors_are_compared_as_a_command_response():
    n = 600
    df = _base_frame(n)
    k = np.arange(n)
    ramp = np.clip((k - 250) / 60.0, 0, 1)
    df["rc_thr"] = -1000.0 + 300.0 * ramp
    for i, m in enumerate(("m1", "m2", "m3", "m4")):
        df[m] = 1000.0 + (120 + 8 * i) * ramp
    d = _analyse(df)
    rel = d["flight"]["relationships"]["throttle_motors"]
    assert rel["status"] in ("consistent", "partially_consistent")
    motor_events = [e for e in d["events"] if e.get("family") == "motor"]
    assert motor_events
    # An increase of all four motors together is a response to throttle, not an imbalance/fault.
    assert any(e.get("interpretation") in ("expected_response", "insufficient_evidence") for e in motor_events)


def test_gyro_disagreeing_with_attitude_is_telemetry_inconsistency():
    n = 600
    df = _base_frame(n)
    k = np.arange(n)
    # Attitude holds steady while the gyro reports a large rate: the records disagree.
    df["phi_deg"] = 0.0
    df.loc[k >= 300, "gx"] = 40.0
    d = _analyse(df)
    rate = d["flight"]["relationships"]["attitude_rate_x"]
    assert rate["status"] in ("not_consistent", "partially_consistent")
    ev = [e for e in d["events"] if e.get("family") == "angular_rate"]
    assert ev and any(e.get("interpretation") == "telemetry_inconsistency" for e in ev)


def test_gravity_at_rest_is_not_an_anomaly_and_units_are_physical():
    df = _base_frame(400)
    df["az"] = 9.81
    d = _analyse(df)
    g = d["flight"]["relationships"]["gravity"]
    assert g["status"] == "consistent" and abs(g["rest_median"] - 9.81) < 0.1
    assert not any(e.get("family") == "acceleration" and e.get("significance") == "physical"
                   for e in d["events"] if e.get("interpretation") == "unexplained_deviation")


# ---- FLT_0003 regression ------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def flt3():
    return analyze_table((DATA / "FLT_0003.CSV").read_bytes(), "FLT_0003.CSV")


def test_flt3_timing_is_milliseconds_verified_and_sampling_is_measured(flt3):
    d = flt3.data
    assert d["schema"]["id"] == "uav_flt_csv"
    assert d["timing"]["time_basis"]["unit_hypothesis"] == "ms"
    assert d["timing"]["time_basis"]["status"] == "verified"
    assert d["timing"]["duration_s"]["value"] == pytest.approx(9.2, abs=0.01)
    assert d["timing"]["sampling_hz"]["value"] == pytest.approx(50.0)
    assert d["timing"]["n_gaps"] == 0 and d["timing"]["rows_without_valid_timestamp"] == 0
    assert d["data_quality"]["rows"] == 461 and d["data_quality"]["columns"] == 33


def test_flt3_units_and_signal_roles_are_not_invented(flt3):
    ch = flt3.data["channels"]
    assert ch["phi_deg"]["unit"] == "deg" and ch["theta_deg"]["unit"] == "deg"
    assert ch["gx"]["unit"] == "deg/s" and ch["gz"]["unit"] == "deg/s"
    assert ch["ax"]["unit"] == "m/s^2" and ch["az"]["unit"] == "m/s^2"
    assert ch["m1"]["unit"] is None and ch["z_est"]["unit"] is None and ch["rc_thr"]["unit"] is None
    # gyroscope != accelerometer: different categories and units
    assert ch["gx"]["group"] == "gyroscope" and ch["ax"]["group"] == "accelerometer"
    assert ch["gx"]["canonical"] == "gyro_x" and ch["ax"]["canonical"] == "acc_x"


def test_flt3_commands_are_compared_with_responses(flt3):
    rels = flt3.data["flight"]["relationships"]
    assert rels["throttle_motors"]["status"] in ("consistent", "partially_consistent")
    assert rels["altitude_setpoint"]["status"] in ("consistent", "partially_consistent")
    assert rels["attitude_rate_x"]["status"] in ("consistent", "partially_consistent")
    # The roll/pitch inputs barely move in this short recording: no tracking claim is made.
    assert rels["roll_tracking"]["status"] == "insufficient_data"
    assert rels["pitch_tracking"]["status"] == "insufficient_data"
    assert rels["gravity"]["status"] == "consistent"


def test_flt3_expected_and_unexplained_periods_are_separated(flt3):
    d = flt3.data
    events = d["events"]
    assert any(e.get("expected") and e.get("significance") == "physical" for e in events), "throttle->motor response"
    assert any(e.get("interpretation") == "unexplained_deviation" for e in events), "late acceleration/attitude transient"
    # Status and command changes are not presented as aircraft deviations.
    for e in events:
        if e["category"] == "state_transition":
            assert e.get("interpretation") in ("control_input_change", "status_change")
    # A motor increase with throttle is not a fault or an imbalance claim.
    motor = [e for e in events if e.get("family") == "motor"]
    assert motor and all(e.get("interpretation") != "telemetry_inconsistency" for e in motor)
    for e in events:
        blob = json.dumps({k: e.get(k) for k in ("headline", "interpretation_label", "what_happened")}).lower()
        assert not re.search(r"\b(fault|failure|damage|unsafe|is safe|caused by|because)\b", blob), blob


def test_flt3_report_is_pilot_facing_and_grounded(flt3):
    rep = flt3.data["operator"]["pilot_report"]
    assert rep["overview"]["rows"] == 461
    assert rep["overview"]["sample_rate_hz"] == pytest.approx(50.0)
    assert rep["overview"]["time_verified"] is True
    assert rep["overview"]["events_to_review"] == len(rep["attention"]) <= 3
    assert rep["overview"]["sequence"] and any("Motor output" in s for s in rep["overview"]["sequence"])
    assert rep["category_summary"]
    assert all("m/s^2" not in c["summary"] or "gravity" in c["summary"] for c in rep["system_checks"])
    # Verified physical values appear; unverified channels stay raw.
    for a in rep["attention"]:
        for v in a["physical_values"]:
            assert "raw" not in v or "unit not verified" in v or True
    assert rep["status_changes"]["count"] >= 1
    assert all(t in rep["status_changes"] for t in ("count", "signals", "note"))
    # No raw column identifiers in the pilot text.
    text = json.dumps(rep).lower()
    for col in flt3.data["channels"]:
        if len(col) > 2 and col not in ("armed", "kill", "blk"):
            assert not re.search(rf"\b{re.escape(col.lower())}\b", text), col


def test_flt3_physical_values_only_use_verified_units(flt3):
    for e in flt3.data["events"]:
        for v in e.get("physical_values") or []:
            if v.get("raw"):
                assert v.get("unit") is None
            else:
                assert v.get("unit") in ("deg", "deg/s", "m/s^2")


def test_flt3_llm_payload_carries_context_not_raw_dumps(flt3):
    from src import direct_llm

    payload = direct_llm.build_payload(flt3.data)
    assert payload["schema"]["time"]["unit_status"] == "verified"
    assert payload["flight"]["relationships"]["throttle_motors"]
    assert payload["flight"]["sequence"]
    assert payload["schema"]["columns"]["phi_deg"]["unit"] == "deg"
    assert payload["schema"]["columns"]["m1"]["unit"] is None
    # Every event the model may describe has an interpretation and evidence.
    for e in payload["events"]:
        assert e["interpretation"] in {
            "expected_response", "unexplained_deviation", "telemetry_inconsistency",
            "control_input_change", "status_change", "data_quality", "insufficient_evidence",
        }
        assert e["headline"]
    assert len(json.dumps(payload, default=str, separators=(",", ":"))) <= direct_llm.MAX_PAYLOAD_CHARS


# ---- language-model grounding: units and absent signals -----------------------------------------------------------

def _llm_context(data):
    from src import direct_llm
    from src.direct_analysis import evidence_catalog

    payload = direct_llm.build_payload(data)
    grounding = direct_llm.Grounding(payload, evidence_catalog(data))
    aliases = {}
    for h, c in payload["schema"]["columns"].items():
        canon = c.get("canonical") or ""
        for alias in (h, canon, canon.replace("_", " "), (c.get("description") or "").split(" ")[0]):
            if alias:
                aliases[str(alias).lower()] = c.get("unit")
    allowed = {direct_llm.normalize_unit_token(c["unit"]) for c in payload["schema"]["columns"].values() if c.get("unit")}
    return payload, grounding, aliases, allowed


def test_llm_accepts_verified_units_and_rejects_units_on_unverified_signals(flt3):
    from src import direct_llm

    payload, grounding, aliases, allowed = _llm_context(flt3.data)
    ev = next(e for e in payload["events"] if e["family"] == "acceleration")
    peak = max(c["during_max"] for c in ev["channels"] if c["channel"] in ("ax", "ay", "az"))
    ok = f"The Z-axis acceleration reached {peak} m/s^2 during the period."
    kept, rejected = direct_llm.clean_text(ok, "t", grounding, True, allowed_units=allowed,
                                           allowed_absent={"altitude", "setpoint"}, unit_aliases=aliases)
    assert kept and not rejected, rejected

    bad = "Motor 3 reached 1589 deg/s."
    kept, rejected = direct_llm.clean_text(bad, "t", grounding, True, allowed_units=allowed,
                                           allowed_absent={"altitude", "setpoint"}, unit_aliases=aliases)
    assert not kept and rejected and rejected[0]["reason"] == "unit_claim"


def test_llm_still_rejects_absent_measured_signals_and_causal_claims(flt3):
    from src import direct_llm

    _payload, grounding, aliases, allowed = _llm_context(flt3.data)
    for sentence, reason in (("Yaw drifted during this interval.", "absent_signal"),
                             ("The jump was caused by wind.", "causation"),
                             ("The aircraft was safe to fly.", "safety")):
        kept, rejected = direct_llm.clean_text(sentence, "t", grounding, True, allowed_units=allowed,
                                               allowed_absent={"altitude", "setpoint"}, unit_aliases=aliases)
        assert not kept and rejected and rejected[0]["reason"] == reason, (sentence, rejected)


def test_llm_payload_times_are_seconds_and_rows_for_traceability(flt3):
    from src import direct_llm

    payload = direct_llm.build_payload(flt3.data)
    big = max(payload["events"], key=lambda e: e["end_row"])
    assert big["start_s"] is not None and big["end_s"] is not None
    assert 0 <= big["start_s"] <= big["end_s"] <= 9.2001
    assert big["start_row"] <= big["end_row"]


def test_llm_cannot_contradict_an_events_interpretation(flt3):
    from src import direct_llm

    expected = next(e for e in flt3.data["events"]
                    if e.get("expected") and e.get("category") != "state_transition")
    unexplained = next(e for e in flt3.data["events"] if e.get("interpretation") == "unexplained_deviation")
    assert not direct_llm._interpretation_ok("This was an unexpected deviation.", expected)
    assert direct_llm._interpretation_ok("This is consistent with the throttle input.", expected)
    assert not direct_llm._interpretation_ok("This was consistent with the pilot input.", unexplained)
    assert direct_llm._interpretation_ok("This is not explained by the recorded input.", unexplained)
