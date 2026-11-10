"""Evidence behind the profile's 'supported' facts, the audited confirmation log, and what follows."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import combined_ingest as ci
from src.combined_preview import build_combined_preview
from src.dataset_profiles import apply_confirmations, fact_digest, load_base_profiles, load_profiles

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "fixtures" / "data"
SHIPPED = ROOT / "config" / "dataset_profiles.json"
G = 9.80665


def _physics():
    spec = importlib.util.spec_from_file_location("analyze_flt_physics", ROOT / "scripts" / "analyze_flt_physics.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("UAV_PROFILES_FILE", str(SHIPPED))
    monkeypatch.setenv("UAV_PROFILE_CONFIRMATIONS", str(tmp_path / "confirmations.json"))


@pytest.fixture(scope="module")
def evidence():
    m = _physics()
    return {n: m.analyze(pd.read_csv(DATA / n)) for n in ("FLT_0001.CSV", "FLT_0002.CSV")}


# ---- the measurements the profile's evidence strings quote -------------------------------------------------

def test_accelerometer_norm_matches_1g_in_si_units(evidence):
    for r in evidence.values():
        assert abs(r["accel_norm"]["ratio_to_g"] - 1.0) < 0.01


def test_each_file_has_exactly_one_stable_rest_segment_and_only_one_is_tilted(evidence):
    r1, r2 = (evidence[n]["rest"] for n in ("FLT_0001.CSV", "FLT_0002.CSV"))
    assert len(r1) == len(r2) == 1 and r1[0]["motors_idle"] and r2[0]["motors_idle"]
    assert r1[0]["phi_std"] < 0.02 and r1[0]["theta_std"] < 0.02
    assert abs(r1[0]["phi_deg"]) > 4 and abs(r1[0]["theta_deg"]) < 1.5  # the whole polarity evidence is ONE tilt, pitch ~1 deg
    assert abs(r2[0]["phi_deg"]) < 0.1 and abs(r2[0]["theta_deg"]) < 0.1


def test_rest_levels_match_the_degree_projection_and_reject_radians(evidence):
    r = evidence["FLT_0001.CSV"]["rest"][0]
    for i, k in enumerate(("ax", "ay", "az")):
        assert r["accel_mean"][i] == pytest.approx(r["predicted_if_degrees"][k], abs=0.03)
    assert abs(r["accel_mean"][1] - r["predicted_if_radians"]["ay"]) > 3
    assert abs(r["accel_mean"][0] - r["predicted_if_radians"]["ax"]) > 3


def test_gyro_roll_gain_is_bracketed_only_under_ms_and_a_wide_band(evidence):
    for name, r in evidence.items():
        ms = r["gyro_vs_angle_rate"]["t_ms=ms"]
        lo = min(min(ms[f"dphi/dt~gx@{k}"]["ols"], ms[f"dphi/dt~gx@{k}"]["reverse"]) for k in (5, 15, 25, 50))
        hi = max(max(ms[f"dphi/dt~gx@{k}"]["ols"], ms[f"dphi/dt~gx@{k}"]["reverse"]) for k in (5, 15, 25, 50))
        assert 0.7 < lo < 1.0 < hi < 1.8, name  # gain ~1 is inside the errors-in-variables band, not pinned to it
        for other in ("t_ms=us", "t_ms=s"):
            assert not 0.3 < r["gyro_vs_angle_rate"][other]["dphi/dt~gx@15"]["ols"] < 3  # off by 1000x


def test_pitch_gain_is_weaker_and_not_claimed_to_be_one(evidence):
    """The gy-vs-pitch-rate relationship: always positive, but the band is wide and file 1 excludes 1.0."""
    for name, r in evidence.items():
        ms = r["gyro_vs_angle_rate"]["t_ms=ms"]
        for k in (5, 15, 25, 50):
            g = ms[f"dtheta/dt~gy@{k}"]
            assert g["ols"] > 0 and g["reverse"] > 0 and g["r"] > 0.6  # sign is clear
    f1 = evidence["FLT_0001.CSV"]["gyro_vs_angle_rate"]["t_ms=ms"]
    assert max(f1[f"dtheta/dt~gy@{k}"]["reverse"] for k in (5, 15, 25, 50)) < 1.0  # gain 1 is outside the band in FLT_0001
    prof = load_profiles()[0]
    assert prof.columns["gy"].unit_status == "supported"  # never 'verified' without the owner
    assert "NOT established better than about +/-40%" in prof.columns["gy"].evidence


def test_height_channels_are_rangefinder_heights_not_barometric(evidence):
    for r in evidence.values():
        h = r["height_channels"]
        assert h["tof_h~tof_mm"]["ols"] == pytest.approx(0.001, rel=0.01) and h["tof_h~tof_mm"]["r"] > 0.999
        assert 0.95 < h["z_est~tof_h"]["ols"] < 1.05 and h["z_est~tof_h"]["r"] > 0.99


def test_sampling_regularity_alone_is_not_the_evidence(evidence):
    for r in evidence.values():
        assert r["time"]["median_step"] == 20.0 and "cannot reveal the unit" in r["time"]["note"]
    assert "NOT used as evidence" in load_profiles()[0].time_evidence


def test_numbers_quoted_in_the_profile_come_from_the_script(evidence):
    """Guard against hand-typed evidence drifting from what the script measures."""
    ev = load_profiles()[0].columns["ay"].evidence
    r = evidence["FLT_0001.CSV"]["rest"][0]
    assert f"{r['accel_mean'][1]:+.3f}" in ev and f"{r['predicted_if_degrees']['ay']:+.3f}" in ev
    assert f"phi={r['phi_deg']:.2f}" in ev and f"theta={r['theta_deg']:.2f}" in ev


# ---- confirmation log: versioned, attributable, auditable, scoped ------------------------------------------------

def rec(fact, value, **kw):
    base = load_base_profiles()[0]
    r = {"action": "confirm", "profile": "uav_flt_csv", "version": base.version, "fact": fact, "value": str(value),
         "confirmed_by": "tester", "confirmed_on": "2026-10-20", "recorded_at": "2026-10-20T10:00:00+00:00", "basis": "doc",
         "fact_digest": fact_digest(base, fact)}
    r.update(kw)
    return r


def test_without_confirmation_nothing_supported_is_applied():
    p = load_profiles()[0]
    assert p.time_unit_status == "supported" and p.columns["ax"].unit_status == "supported"
    mapping = ci.propose_mapping(ci.inspect_combined((DATA / "FLT_0001.CSV").read_bytes(), "f"), profile=p)
    assert not any(c.validation_status == ci.VALID for c in mapping.columns if c.canonical)


def test_confirmations_promote_exactly_the_confirmed_facts():
    base = load_base_profiles()[0]
    q = apply_confirmations(base, [rec("time_unit", "ms"), rec("accel_unit", "m/s^2"), rec("az_polarity", -1)])
    assert q.time_unit_status == "verified" and "owner confirmation by tester" in q.time_evidence
    assert all(q.columns[k].unit_status == "verified" for k in ("ax", "ay", "az"))
    assert q.columns["az"].frame_status == "verified" and q.columns["az"].frame_sign == -1
    assert q.columns["ax"].frame_status != "verified"  # needs pitch_sign
    assert q.columns["gx"].unit_status == "supported" and q.columns["gz"].unit_status == "unverified"
    assert len(q.applied_confirmations) == 3


def test_confirming_gyro_unit_does_not_verify_gz():
    q = apply_confirmations(load_base_profiles()[0], [rec("gyro_unit", "deg/s")])
    assert q.columns["gx"].unit_status == q.columns["gy"].unit_status == "verified"
    assert q.columns["gz"].unit_status == "unverified"  # no evidence, own fact


def test_a_frame_is_verified_only_when_every_choice_it_uses_is_confirmed():
    base = load_base_profiles()[0]
    q = apply_confirmations(base, [rec("roll_sign", 1)])
    assert q.columns["ay"].frame_status == "verified" and q.columns["ay"].frame_sign == -1
    assert q.columns["ax"].frame_status != "verified" and q.columns["gy"].frame_status != "verified"


@pytest.mark.parametrize("bad,fragment", [
    (dict(fact="time_unit", value="us"), "not an allowed value"),
    (dict(fact="roll_sign", value="0"), "not an allowed value"),
    (dict(fact="az_polarity", value="1"), "not an allowed value"),
    (dict(fact="time_unit", value="ms", version="0.9"), "version"),
    (dict(fact="time_unit", value="ms", fact_digest="0" * 64), "definition or evidence changed"),
    (dict(fact="time_unit", value="ms", confirmed_by=""), "lacks an author or a basis"),
    (dict(fact="time_unit", value="ms", basis="  "), "lacks an author or a basis"),
])
def test_invalid_stale_or_anonymous_records_are_ignored_and_reported(bad, fragment):
    base = load_base_profiles()[0]
    entry = rec(bad.get("fact", "time_unit"), bad.get("value", "ms"))
    entry.update({k: v for k, v in bad.items() if k not in ("fact", "value")})
    entry.update(fact=bad["fact"], value=bad["value"])
    if "fact_digest" not in bad:
        entry["fact_digest"] = fact_digest(base, bad["fact"])
    q = apply_confirmations(base, [entry])
    assert q.applied_confirmations == [] and q.time_unit_status == "supported"
    assert any(fragment in n for n in q.confirmation_notes), q.confirmation_notes


def test_editing_the_evidence_invalidates_an_earlier_confirmation(tmp_path):
    base = load_base_profiles()[0]
    old = rec("accel_unit", "m/s^2")
    data = json.loads(SHIPPED.read_text())
    data["profiles"][0]["columns"]["ax"]["evidence"] += " (edited)"
    p = tmp_path / "edited.json"
    p.write_text(json.dumps(data))
    edited = load_base_profiles(p)[0]
    q = apply_confirmations(edited, [old])
    assert q.columns["ax"].unit_status == "supported" and any("changed after it was recorded" in n for n in q.confirmation_notes)


def test_latest_record_wins_and_revoke_removes_it():
    base = load_base_profiles()[0]
    first = rec("roll_sign", 1, recorded_at="2026-10-20T10:00:00+00:00")
    second = rec("roll_sign", -1, recorded_at="2026-10-21T10:00:00+00:00")
    q = apply_confirmations(base, [first, second])
    assert q.columns["phi_deg"].frame_sign == -1 and q.columns["ay"].frame_sign == 1
    q = apply_confirmations(base, [first, second, rec("roll_sign", "", action="revoke", fact_digest=None)])
    assert q.columns["phi_deg"].frame_status != "verified"


def test_confirmations_are_scoped_to_their_own_profile():
    base = load_base_profiles()[0]
    other = rec("time_unit", "ms", profile="some_other_profile")
    q = apply_confirmations(base, [other])
    assert q.applied_confirmations == [] and q.time_unit_status == "supported" and q.confirmation_notes == []


def test_overlay_file_is_read_and_the_profile_file_is_untouched(tmp_path, monkeypatch):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"confirmations": [rec("time_unit", "ms")]}))
    monkeypatch.setenv("UAV_PROFILE_CONFIRMATIONS", str(path))
    before = SHIPPED.read_bytes()
    assert load_profiles()[0].time_unit_status == "verified"
    assert SHIPPED.read_bytes() == before


def test_confirmations_never_touch_three_file_uploads_or_the_detector(tmp_path, monkeypatch):
    """A confirmation log changes only combined-file profile mapping."""
    from src import inference_service as inf
    from src.upload import load_individual_csvs
    from tests.conftest import _make_att, _make_baro, _make_imu

    def run():
        files = {s: df.to_csv(index=False).encode() for s, df in
                 (("IMU", _make_imu(500)), ("ATT", _make_att(200)), ("BARO", _make_baro(200)))}
        up = load_individual_csvs(files)
        assert up.ok
        inf.clear_caches()
        r = inf.analyze_flight(up.frames, flight_id="x", model_dir=str(ROOT / "uav_model"))
        inf.clear_caches()
        return r

    before = run()
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"confirmations": [rec(f, v) for f, v in
                                                  (("time_unit", "ms"), ("accel_unit", "m/s^2"), ("roll_sign", 1), ("pitch_sign", 1))]}))
    monkeypatch.setenv("UAV_PROFILE_CONFIRMATIONS", str(path))
    after = run()
    assert np.array_equal(before.windows.anomaly_score.values, after.windows.anomaly_score.values)
    assert before.threshold == after.threshold == 4.034964084625244 and before.features == after.features


def _cli(*args, conf_path):
    env = {**os.environ, "UAV_PROFILE_CONFIRMATIONS": str(conf_path), "UAV_PROFILES_FILE": str(SHIPPED)}
    return subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "confirm_profile_facts.py"), *args],
                          capture_output=True, text=True, env=env, cwd=ROOT)


def test_cli_audit_trail(tmp_path):
    target = tmp_path / "conf.json"
    r = _cli("--list", conf_path=target)
    assert r.returncode == 0 and "roll_sign in [1, -1]" in r.stdout and "if -1:" in r.stdout and "how to check" in r.stdout
    assert _cli("--confirm", "time_unit=ms", "--by", "A", "--basis", "x", conf_path=target).stdout.count("Dry run") == 1
    assert not target.exists()
    assert _cli("--confirm", "time_unit=us", "--by", "A", "--basis", "x", "--yes", conf_path=target).returncode == 2
    assert _cli("--confirm", "roll_sign=0", "--by", "A", "--basis", "x", "--yes", conf_path=target).returncode == 2
    assert _cli("--confirm", "time_unit=ms", "--by", "A", "--yes", conf_path=target).returncode == 2  # basis mandatory
    assert _cli("--confirm", "time_unit=ms", "--basis", "x", "--yes", conf_path=target).returncode == 2  # author mandatory
    assert not target.exists()
    ok = _cli("--confirm", "time_unit=ms", "roll_sign=-1", "--by", "A", "--basis", "tilt test", "--yes", conf_path=target)
    assert ok.returncode == 0
    saved = json.loads(target.read_text())["confirmations"]
    assert [e["fact"] for e in saved] == ["time_unit", "roll_sign"]
    assert all(e["confirmed_by"] == "A" and e["basis"] == "tilt test" and e["recorded_at"] and len(e["fact_digest"]) == 64 and e["version"] == "1.2.0"
               for e in saved)
    assert "already confirmed" in _cli("--confirm", "time_unit=ms", "--by", "A", "--basis", "x", "--yes", conf_path=target).stdout
    assert _cli("--confirm", "roll_sign=1", "--by", "A", "--basis", "x", "--yes", conf_path=target).returncode == 2  # needs --supersede
    assert _cli("--confirm", "roll_sign=1", "--by", "B", "--basis", "recheck", "--supersede", "--yes", conf_path=target).returncode == 0
    assert _cli("--revoke", "time_unit", "--by", "B", "--basis", "wrong", "--yes", conf_path=target).returncode == 0
    log = json.loads(target.read_text())["confirmations"]
    assert [(e["action"], e["fact"]) for e in log] == [("confirm", "time_unit"), ("confirm", "roll_sign"), ("confirm", "roll_sign"), ("revoke", "time_unit")]
    assert log[2]["supersedes"] == 1  # nothing was overwritten
    h = _cli("--history", conf_path=target).stdout
    assert "#2" in h and "roll_sign=1" in h and "revoke" in h


# ---- what follows -------------------------------------------------------------------------------------------------------

ALL = [("time_unit", "ms"), ("angle_unit", "deg"), ("accel_unit", "m/s^2"), ("gyro_unit", "deg/s"), ("gyro_z_unit", "deg/s"),
       ("az_polarity", -1), ("roll_sign", 1), ("pitch_sign", -1), ("gyro_z_sign", 1)]


@pytest.fixture()
def fully_confirmed(tmp_path, monkeypatch):
    path = tmp_path / "all.json"
    path.write_text(json.dumps({"confirmations": [rec(f, v) for f, v in ALL]}))
    monkeypatch.setenv("UAV_PROFILE_CONFIRMATIONS", str(path))


@pytest.mark.parametrize("name", ["FLT_0001.CSV", "FLT_0002.CSV"])
def test_fully_confirmed_real_files_convert_per_axis_but_still_cannot_be_scored(fully_confirmed, name):
    insp = ci.inspect_combined((DATA / name).read_bytes(), name)
    m = ci.propose_mapping(insp, profile=load_profiles()[0])
    assert all(c.validation_status == ci.VALID for c in m.columns if c.canonical)
    frames = ci.to_stream_frames(insp, m)
    raw = pd.read_csv(DATA / name)
    imu, att = frames["IMU"].reset_index(drop=True), frames["ATT"].reset_index(drop=True)
    # (roll_sign, pitch_sign) = (+1, -1): ROS-style. ax unchanged, ay and az negated, pitch and gy negated.
    np.testing.assert_allclose(imu["AccX"], raw["ax"], rtol=1e-12)
    np.testing.assert_allclose(imu["AccY"], -raw["ay"], rtol=1e-12)
    np.testing.assert_allclose(imu["AccZ"], -raw["az"], rtol=1e-12)
    np.testing.assert_allclose(imu["GyrX"], raw["gx"] * math.pi / 180, rtol=1e-12)
    np.testing.assert_allclose(imu["GyrY"], -raw["gy"] * math.pi / 180, rtol=1e-12)
    np.testing.assert_allclose(att["Roll"], raw["phi_deg"], rtol=1e-12)
    np.testing.assert_allclose(att["Pitch"], -raw["theta_deg"], rtol=1e-12)
    p = build_combined_preview((DATA / name).read_bytes(), name, {}, False)
    a = p["assessment"]
    assert a["outcome"] == "incompatible" and a["unresolved"] == []
    assert {x["canonical"] for x in a["missing"]} == {"yaw", "des_roll", "des_pitch"}
    assert [s["kind"] for s in a["next_steps"]] == ["log_signals"]


def test_before_confirmation_next_steps_separate_by_what_can_settle_them():
    p = build_combined_preview((DATA / "FLT_0001.CSV").read_bytes(), "f", {}, False)
    steps = {s["kind"]: s for s in p["assessment"]["next_steps"]}
    assert set(steps) == {"confirm", "tilt_test", "document", "log_signals"}
    assert set(steps["confirm"]["facts"]) == {"time_unit", "angle_unit", "accel_unit", "gyro_unit", "az_polarity"}
    assert set(steps["tilt_test"]["facts"]) == {"roll_sign", "pitch_sign"}
    assert "raise the NOSE" in steps["tilt_test"]["text"] and "lower the RIGHT wing" in steps["tilt_test"]["text"]
    assert set(steps["document"]["facts"]) == {"gyro_z_unit", "gyro_z_sign"}
