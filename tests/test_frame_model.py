"""Frame / polarity model for the FLT layout, derived from the relations the data must satisfy.

Training (model) convention T: AccX' = +g sin(th'), AccY' = -g sin(ph') cos(th'), AccZ' = -g cos(ph')cos(th'),
roll rate = GyrX', pitch rate = GyrY'. The FLT file shows, in its own angles (ph_f, th_f):
    ax_f = -g sin(th_f),  ay_f = +g sin(ph_f) cos(th_f),  az_f = +g cos(ph_f) cos(th_f),
    d(ph_f)/dt = +gx_f,   d(th_f)/dt = +gy_f.
If ph' = s_roll * ph_f-convention and th' likewise, those relations force
    az' = -az_f, ay' = -s_roll*ay_f, ax' = -s_pitch*ax_f, gx' = s_roll*gx_f, gy' = s_pitch*gy_f,
    roll' = s_roll*ph_f, pitch' = s_pitch*th_f.
Rest data cannot choose s_roll / s_pitch (all four combinations reproduce it), so they are owner facts.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import combined_ingest as ci
from src.dataset_profiles import load_profiles

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "fixtures" / "data"
SHIPPED = ROOT / "config" / "dataset_profiles.json"
G = 9.80665


def conf(fact, value):
    return {"action": "confirm", "profile": "uav_flt_csv", "version": "1.2.0", "fact": fact, "value": str(value),
            "confirmed_by": "tester", "confirmed_on": "2026-10-20", "recorded_at": "2026-10-20T00:00:00Z", "basis": "test"}


def digest_for(fact):
    from src.dataset_profiles import fact_digest, load_base_profiles

    return fact_digest(load_base_profiles()[0], fact)


def conf_d(fact, value):
    return {**conf(fact, value), "fact_digest": digest_for(fact)}


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("UAV_PROFILES_FILE", str(SHIPPED))
    monkeypatch.setenv("UAV_PROFILE_CONFIRMATIONS", str(tmp_path / "c.json"))


def write_conf(tmp_path, monkeypatch, s_roll, s_pitch, s_gz=1):
    entries = [conf_d(f, v) for f, v in (("time_unit", "ms"), ("angle_unit", "deg"), ("accel_unit", "m/s^2"),
                                          ("gyro_unit", "deg/s"), ("gyro_z_unit", "deg/s"), ("az_polarity", -1),
                                          ("roll_sign", s_roll), ("pitch_sign", s_pitch), ("gyro_z_sign", s_gz))]
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"confirmations": entries}))
    monkeypatch.setenv("UAV_PROFILE_CONFIRMATIONS", str(p))


def truth(n=400, seed=0):
    """Model-convention (FRD) ground truth: attitude, rates, and the specific force they imply at rest."""
    rng = np.random.default_rng(seed)
    ph = rng.uniform(-8, 8, n)
    th = rng.uniform(-8, 8, n)
    r, p = np.deg2rad(ph), np.deg2rad(th)
    a = G * np.stack([np.sin(p), -np.sin(r) * np.cos(p), -np.cos(r) * np.cos(p)], axis=1)
    w = rng.normal(0, 20, (n, 3))  # deg/s
    return ph, th, a, w


def make_file(s_roll, s_pitch, s_gz, seed=0):
    """What the FLT logger would write if its roll/pitch signs relative to the model were (s_roll, s_pitch)."""
    ph, th, a, w = truth(seed=seed)
    f = pd.DataFrame({"t_ms": np.arange(len(ph)) * 20})
    f["phi_deg"], f["theta_deg"] = s_roll * ph, s_pitch * th
    f["ax"], f["ay"], f["az"] = -s_pitch * a[:, 0], -s_roll * a[:, 1], -a[:, 2]
    f["gx"], f["gy"], f["gz"] = s_roll * w[:, 0], s_pitch * w[:, 1], s_gz * w[:, 2]
    # the file's own observed relations, as measured in FLT_0001 (guards the generator itself)
    assert np.allclose(f.ax, -G * np.sin(np.deg2rad(f.theta_deg)))
    assert np.allclose(f.ay, G * np.sin(np.deg2rad(f.phi_deg)) * np.cos(np.deg2rad(f.theta_deg)))
    assert np.allclose(f.az, G * np.cos(np.deg2rad(f.phi_deg)) * np.cos(np.deg2rad(f.theta_deg)))
    for c in ("rc_roll", "rc_pitch", "rc_thr", "rc_yaw", "armed"):
        f[c] = 0
    return f, (ph, th, a, w)


def csv(d):
    import io

    b = io.BytesIO()
    d.to_csv(b, index=False)
    return b.getvalue()


@pytest.mark.parametrize("s_roll,s_pitch", list(itertools.product([1, -1], [1, -1])))
def test_every_convention_recovers_the_model_frame_exactly(tmp_path, monkeypatch, s_roll, s_pitch):
    write_conf(tmp_path, monkeypatch, s_roll, s_pitch)
    f, (ph, th, a, w) = make_file(s_roll, s_pitch, 1)
    insp = ci.inspect_combined(csv(f), "f.csv")
    prof = load_profiles()[0]
    m = ci.propose_mapping(insp, profile=prof)
    by = {c.source_column: c for c in m.columns}
    assert all(by[k].validation_status == ci.VALID for k in ("phi_deg", "theta_deg", "ax", "ay", "az", "gx", "gy", "gz"))
    fr = ci.to_stream_frames(insp, m)
    imu, att = fr["IMU"].reset_index(drop=True), fr["ATT"].reset_index(drop=True)
    np.testing.assert_allclose(att["Roll"], ph, atol=1e-9)
    np.testing.assert_allclose(att["Pitch"], th, atol=1e-9)
    for i, col in enumerate(("AccX", "AccY", "AccZ")):
        np.testing.assert_allclose(imu[col], a[:, i], atol=1e-9, err_msg=col)
    for i, col in enumerate(("GyrX", "GyrY", "GyrZ")):
        np.testing.assert_allclose(imu[col], np.deg2rad(w[:, i]), atol=1e-12, err_msg=col)


def test_the_old_uniform_accelerometer_flip_is_wrong_for_three_of_four_conventions(tmp_path, monkeypatch):
    """Guards against re-introducing 'negate every accelerometer axis because az is +9.8'."""
    wrong = []
    for s_roll, s_pitch in itertools.product([1, -1], [1, -1]):
        _f, (ph, th, a, w) = make_file(s_roll, s_pitch, 1)
        f = make_file(s_roll, s_pitch, 1)[0]
        old = -f[["ax", "ay", "az"]].to_numpy()  # uniform flip, angles and gyros untouched
        ok = (np.allclose(old, a, atol=1e-9) and np.allclose(f.phi_deg, ph) and np.allclose(f.theta_deg, th)
              and np.allclose(f.gx, w[:, 0]) and np.allclose(f.gy, w[:, 1]))
        if not ok:
            wrong.append((s_roll, s_pitch))
    assert wrong == [(1, -1), (-1, 1), (-1, -1)]  # right only when both angle conventions already match training


def test_signs_are_derived_not_proposed(tmp_path, monkeypatch):
    write_conf(tmp_path, monkeypatch, 1, -1)  # ROS-style: right-handed about a left-pointing y axis => nose-down positive
    p = load_profiles()[0]
    sg = {k: p.columns[k].frame_sign for k in ("phi_deg", "theta_deg", "ax", "ay", "az", "gx", "gy", "gz")}
    assert sg == {"phi_deg": 1, "theta_deg": -1, "ax": 1, "ay": -1, "az": -1, "gx": 1, "gy": -1, "gz": 1}


def test_without_owner_choices_no_frame_is_applied_and_nothing_is_proposed():
    p = load_profiles()[0]
    for k in ("phi_deg", "theta_deg", "ax", "ay", "az", "gx", "gy", "gz"):
        assert p.columns[k].frame_status != "verified", k
    assert "roll_sign" in p.choices and "pitch_sign" in p.choices
    assert p.choices["roll_sign"]["values"] == [1, -1] and p.choices["az_polarity"]["values"] == [-1]
    insp = ci.inspect_combined((DATA / "FLT_0001.CSV").read_bytes(), "f")
    m = ci.propose_mapping(insp, profile=p)
    assert not any(c.validation_status == ci.VALID for c in m.columns if c.canonical)
