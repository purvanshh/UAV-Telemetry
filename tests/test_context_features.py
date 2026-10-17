"""Attitude-rate relationship, command tracking and coarse phase."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.context_features import (
    ACTIVE_RAD_S, VERTICAL_MPS, derive_signals, verify_rate_convention, window_phase,
)


def _aligned(n=600, seed=0, big_roll=False, gap=False):
    t = 100 + np.arange(n) * 0.1
    rng = np.random.default_rng(seed)
    # smooth random-ish attitude in radians
    def smooth(a):
        return np.convolve(rng.normal(0, 1, n + 20), np.ones(21) / 21, mode="valid")[:n] * a
    phi = smooth(2.0 if big_roll else 0.6)
    th = smooth(0.8)
    psi = np.cumsum(smooth(0.4)) * 0.1 + 3.0  # drifts across the ±π wrap
    dphi, dth, dpsi = (np.gradient(x, t) for x in (phi, th, psi))
    p = dphi - dpsi * np.sin(th)
    q = dth * np.cos(phi) + dpsi * np.cos(th) * np.sin(phi)
    r = -dth * np.sin(phi) + dpsi * np.cos(th) * np.cos(phi)
    sid = np.zeros(n, dtype=int)
    if gap:
        sid[n // 2 - 2 : n // 2 + 2] = -1
        sid[n // 2 + 2 :] = 1
    df = pd.DataFrame({
        "_t_s": t, "_session_id": sid,
        "gyro_x": p, "gyro_y": q, "gyro_z": r,
        "roll": np.rad2deg(phi), "pitch": np.rad2deg(th),
        "yaw_sin": np.sin(psi), "yaw_cos": np.cos(psi),
        "roll_res": 0.5, "pitch_res": -0.5,
        "alt": np.linspace(0, 6, n),
    })
    return df, (p, q, r)


def test_zyx_rates_match_gyro_when_consistent_including_yaw_wrap():
    a, (p, q, r) = _aligned()
    s = derive_signals(a)
    for ax, g in zip("xyz", (p, q, r)):
        res = s[f"rate_res_{ax}"].to_numpy()[5:-5]
        assert np.nanmax(np.abs(res)) < 0.05 * max(1.0, np.abs(g).max())  # not distorted by the ±π wrap


def test_rotation_matters_for_large_roll():
    a, (p, q, r) = _aligned(big_roll=True)
    s = derive_signals(a)
    zyx = np.nanmean(np.abs(s["rate_res_y"][5:-5]))
    naive = np.nanmean(np.abs(s["rate_naive_y"][5:-5] - q[5:-5]))
    assert zyx < naive


def test_inconsistent_gyro_is_visible_in_residual():
    a, _ = _aligned()
    clean = derive_signals(a)["rate_res_x"].abs().max()
    a.loc[200:230, "gyro_x"] += 1.0
    s = derive_signals(a)
    assert s["rate_res_x"].abs().iloc[200:230].mean() > 10 * clean


def test_no_differencing_across_gap():
    a, _ = _aligned(gap=True)
    s = derive_signals(a)
    gap = a["_session_id"] < 0
    assert s.loc[gap, "rate_res_x"].isna().all()


def test_verification_gate_passes_consistent_and_fails_random():
    a, _ = _aligned()
    good = verify_rate_convention({"f": derive_signals(a)}, {"f": a}, gating=["f"])
    assert good["verified"] and all(r["zyx"]["corr"] > 0.99 for r in good["rows"])
    b = a.copy()
    b[["gyro_x", "gyro_y", "gyro_z"]] = np.random.default_rng(3).normal(0, 1, (len(b), 3))
    bad = verify_rate_convention({"f": derive_signals(b)}, {"f": b}, gating=["f"])
    assert not bad["verified"]
    assert not verify_rate_convention({"f": derive_signals(a)}, {"f": a}, gating=[])["verified"]


def test_phase_rule():
    v = np.array([VERTICAL_MPS, 0.0, 0.0, 0.0])
    g = np.array([0.0, ACTIVE_RAD_S, 0.0, 0.05])
    assert list(window_phase(v, g)) == ["vertical", "active", "steady", "steady"]


def test_command_yaw_residual_wraps(tmp_path):
    a, _ = _aligned(n=100)
    att = pd.DataFrame({"__t_s": a["_t_s"], "Yaw": 179.0, "DesYaw": -179.0})  # 2° apart across the wrap
    s = derive_signals(a, att, lambda ts, y, tg, mg: np.interp(tg, ts, y), 0.5)
    assert np.allclose(s["cmd_yaw"], 2.0)
