"""Context / consistency signals derived from the aligned 10 Hz frame.

Nothing here feeds the deployed LSTM (its 12-feature input is fixed). These
signals back *side detectors* evaluated in ``src/alt_eval.py``.

Conventions (and how they are verified)
---------------------------------------
* ATT ``Roll/Pitch/Yaw`` are degrees, ZYX (yaw-pitch-roll) Euler angles; IMU
  ``Gyr*`` are body-frame rates in rad/s (units as declared by the schema).
* Euler rates → body rates (ZYX)::

      p = φ̇ − ψ̇ sinθ
      q = θ̇ cosφ + ψ̇ cosθ sinφ
      r = −θ̇ sinφ + ψ̇ cosθ cosφ

  φ̇, θ̇, ψ̇ are central differences of the 10 Hz attitude within a session
  (yaw unwrapped within a session; never differenced across a gap).
* :func:`verify_rate_convention` compares derived rates with the gyro on real,
  un-perturbed flights (correlation and regression slope, per axis). The
  consistency feature is only used when every real flight clears the gate;
  otherwise it is reported as *unverified* and excluded. This check uses no
  anomaly labels.
* ``alt_rate`` in the aligned frame is a difference per **grid step** (0.1 s),
  not per second — it does not divide by dt even though the schema says m/s.
  It is not scored by the model. Phase context here recomputes vertical speed
  as ``d(alt)/dt`` in m/s.

Coarse flight phase (per window, fixed a priori — not tuned):
``vertical`` if |net altitude change over the window| / duration ≥ 0.5 m/s
(net change, not the mean of 10 Hz differences: differencing 10 Hz BARO is noise-dominated
and labelled ~88% of one real flight "vertical" in a first draft of this rule — corrected
because of that phase distribution, before any detector comparison was re-run); else ``active`` if gyro RMS
(3-axis) ≥ 0.1 rad/s; else ``steady``. BARO ``Alt``'s reference (home vs sea
level) is not relied on — only its rate of change.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

PHASES = ("steady", "active", "vertical")
VERTICAL_MPS = 0.5
ACTIVE_RAD_S = 0.1
VERIFY_MIN_CORR = 0.7  # per axis, on every real flight
VERIFY_MIN_SLOPE, VERIFY_MAX_SLOPE = 0.6, 1.6  # gyro ≈ derived (rad/s both)


def _segments(sid: np.ndarray):
    for s in np.unique(sid):
        if s < 0:
            continue
        idx = np.flatnonzero(sid == s)
        if len(idx) >= 3:
            yield idx


def _grad(x: np.ndarray, t: np.ndarray, sid: np.ndarray) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for idx in _segments(sid):
        out[idx] = np.gradient(x[idx], t[idx])
    return out


def derive_signals(
    aligned: pd.DataFrame,
    att_clean: Optional[pd.DataFrame] = None,
    interp=None,
    max_gap_s: float = 0.5,
) -> pd.DataFrame:
    """Per-grid-sample context signals (NaN where inputs are NaN or in gaps)."""
    t = aligned["_t_s"].to_numpy(dtype=float)
    sid = aligned["_session_id"].to_numpy()
    phi = np.deg2rad(aligned["roll"].to_numpy(dtype=float))
    th = np.deg2rad(aligned["pitch"].to_numpy(dtype=float))
    yaw = np.arctan2(aligned["yaw_sin"].to_numpy(dtype=float), aligned["yaw_cos"].to_numpy(dtype=float))
    yaw_u = yaw.copy()
    for idx in _segments(sid):
        yaw_u[idx] = np.unwrap(yaw[idx])
    dphi, dth, dpsi = _grad(phi, t, sid), _grad(th, t, sid), _grad(yaw_u, t, sid)
    p = dphi - dpsi * np.sin(th)
    q = dth * np.cos(phi) + dpsi * np.cos(th) * np.sin(phi)
    r = -dth * np.sin(phi) + dpsi * np.cos(th) * np.cos(phi)
    gyro = aligned[["gyro_x", "gyro_y", "gyro_z"]].to_numpy(dtype=float)
    out = pd.DataFrame({"_t_s": t, "_session_id": sid})
    for k, (d, g) in enumerate(zip((p, q, r), gyro.T)):
        out[f"rate_der_{'xyz'[k]}"] = d
        out[f"rate_res_{'xyz'[k]}"] = d - g
    # naive (non-rotated) derivative, kept only to show the rotation matters
    out["rate_naive_x"], out["rate_naive_y"], out["rate_naive_z"] = dphi, dth, dpsi
    out["cmd_roll"] = aligned["roll_res"].to_numpy(dtype=float)
    out["cmd_pitch"] = aligned["pitch_res"].to_numpy(dtype=float)
    cmd_yaw = np.full(len(t), np.nan)
    if att_clean is not None and interp is not None:
        cols = {c.lower(): c for c in att_clean.columns}
        if "desyaw" in cols and "yaw" in cols:
            dy = att_clean[cols["desyaw"]].to_numpy(dtype=float)
            yw = att_clean[cols["yaw"]].to_numpy(dtype=float)
            res = (dy - yw + 180.0) % 360.0 - 180.0  # wrapped at the raw sample, then resampled
            cmd_yaw = interp(att_clean["__t_s"].to_numpy(), res, t, max_gap_s)
    out["cmd_yaw"] = cmd_yaw
    out["alt"] = aligned["alt"].to_numpy(dtype=float)
    out["vert_speed"] = _grad(aligned["alt"].to_numpy(dtype=float), t, sid)  # noisy at 10 Hz; windows use net change
    out["gyro_norm"] = np.sqrt((gyro**2).sum(axis=1))
    return out


def window_phase(vert_mean_abs: np.ndarray, gyro_rms: np.ndarray) -> np.ndarray:
    ph = np.where(
        vert_mean_abs >= VERTICAL_MPS, "vertical",
        np.where(gyro_rms >= ACTIVE_RAD_S, "active", "steady"),
    )
    return ph.astype(object)


def verify_rate_convention(
    signals: Mapping[str, pd.DataFrame],
    aligned: Mapping[str, pd.DataFrame],
    gating: Sequence[str],
) -> Dict[str, Any]:
    """Compare derived body rates with the gyro. ``gating`` flights must all pass."""
    rows: List[Dict[str, Any]] = []
    for fid, sg in signals.items():
        a = aligned[fid]
        ok = sg["_session_id"].to_numpy() >= 0
        for k, ax in enumerate("xyz"):
            g = a[f"gyro_{ax}"].to_numpy(dtype=float)
            row: Dict[str, Any] = {"flight_id": fid, "axis": "pqr"[k], "gating": fid in gating}
            for name, col in (("zyx", f"rate_der_{ax}"), ("naive", f"rate_naive_{ax}")):
                d = sg[col].to_numpy(dtype=float)
                v = np.isfinite(d) & np.isfinite(g) & ok
                if v.sum() < 50:
                    row[name] = None
                    continue
                c = float(np.corrcoef(d[v], g[v])[0, 1])
                slope = float(np.polyfit(d[v], g[v], 1)[0])
                row[name] = {"corr": c, "slope": slope, "n": int(v.sum())}
            rows.append(row)
    gate_rows = [r for r in rows if r["gating"] and r["zyx"]]
    passed = bool(gate_rows) and all(
        r["zyx"]["corr"] >= VERIFY_MIN_CORR and VERIFY_MIN_SLOPE <= r["zyx"]["slope"] <= VERIFY_MAX_SLOPE
        for r in gate_rows
    )
    return {
        "verified": passed,
        "gate": f"corr >= {VERIFY_MIN_CORR} and {VERIFY_MIN_SLOPE} <= slope <= {VERIFY_MAX_SLOPE} on every axis of every gating flight",
        "gating_flights": list(gating),
        "rows": rows,
        "reason": None if passed else ("no gating flights" if not gate_rows else "a gating flight failed the gate"),
    }
