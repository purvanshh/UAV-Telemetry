"""Physical-consistency analysis of a flight-log CSV (units / frame / sampling evidence).

This produces EVIDENCE, not documentation. Each test asks what the data would look like under a
hypothesis and reports how well it fits; a fit raises or lowers confidence but never by itself marks
a profile fact ``verified``.

    python scripts/analyze_flt_physics.py tests/fixtures/data/FLT_0001.CSV [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

G = 9.80665


def _fit(y: np.ndarray, x: np.ndarray) -> Dict[str, float]:
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 10 or np.std(x) < 1e-12:
        return {"slope": float("nan"), "r": float("nan"), "n": int(len(x))}
    a, b = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    return {"slope": float(a), "intercept": float(b), "r": r, "n": int(len(x))}


def gain(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Gain of y on x with errors-in-variables bounds.

    OLS (y on x) is biased LOW when x is noisy; the reverse regression (x on y, inverted) is biased
    HIGH. The true gain lies between them (total least squares is in between). A relationship is only
    called 'consistent with gain g' if g lies inside [ols, reverse].
    """
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    b_yx = float(np.polyfit(x, y, 1)[0])
    b_xy = float(np.polyfit(y, x, 1)[0])
    sxx, syy, sxy = np.var(x), np.var(y), np.cov(x, y, bias=True)[0, 1]
    tls = float((syy - sxx + np.sqrt((syy - sxx) ** 2 + 4 * sxy**2)) / (2 * sxy))
    return {"ols": b_yx, "reverse": float(1 / b_xy) if b_xy else float("nan"), "tls": tls,
            "r": float(np.corrcoef(x, y)[0, 1]), "n": int(len(x))}


def rest_segments(df: pd.DataFrame) -> list:
    """Disarmed stretches: the only place gravity projection is a valid test (in flight a multirotor's
    accelerometer mostly measures thrust along body-z, not the tilt of gravity)."""
    idx = np.where(df["armed"].to_numpy() == 0)[0]
    if not len(idx):
        return []
    runs = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
    out = []
    for r in runs:
        if len(r) < 50:
            continue
        s = df.iloc[r]
        a = s[["ax", "ay", "az"]].to_numpy(float)
        ph, th = s["phi_deg"].mean(), s["theta_deg"].mean()
        pr, tr = np.deg2rad(ph), np.deg2rad(th)
        out.append({
            "rows": [int(r[0]), int(r[-1])], "n": int(len(r)),
            "motors_idle": bool((s[["m1", "m2", "m3", "m4"]].max().max() <= 1010)),
            "phi_deg": float(ph), "theta_deg": float(th),
            "phi_std": float(s["phi_deg"].std()), "theta_std": float(s["theta_deg"].std()),
            "accel_mean": a.mean(axis=0).tolist(), "accel_std": a.std(axis=0).tolist(),
            "norm_mean": float(np.linalg.norm(a, axis=1).mean()),
            "gyro_mean": s[["gx", "gy", "gz"]].mean().tolist(),
            "predicted_if_degrees": {"ax": float(-G * np.sin(tr)), "ay": float(G * np.sin(pr) * np.cos(tr)),
                                     "az": float(G * np.cos(pr) * np.cos(tr))},
            "predicted_if_radians": {"ax": float(-G * np.sin(th)), "ay": float(G * np.sin(ph) * np.cos(th))},
        })
    return out


def analyze(df: pd.DataFrame, time_col: str = "t_ms") -> Dict[str, Any]:
    out: Dict[str, Any] = {"rows": int(len(df))}
    t = df[time_col].to_numpy(float)
    dt = np.diff(t)
    out["time"] = {
        "median_step": float(np.median(dt)), "min_step": float(dt.min()), "max_step": float(dt.max()),
        "monotonic": bool((dt > 0).all()), "span": float(t[-1] - t[0]),
        "note": "A steady step shows regular sampling; it cannot reveal the unit.",
    }
    a = df[["ax", "ay", "az"]].to_numpy(float)
    norm = np.linalg.norm(a, axis=1)
    out["accel_norm"] = {
        "mean": float(norm.mean()), "median": float(np.median(norm)), "std": float(norm.std()),
        "ratio_to_g": float(np.median(norm) / G),
        "reading": "median |a| close to 9.8 means the three axes are in m/s^2 (a 'g' unit would give ~1.0)",
    }
    out["rest"] = rest_segments(df)
    out["rest_note"] = ("Gravity projection is evaluated only on disarmed segments. Each file has a single rest "
                        "segment; polarity of ax rests on one tilt of ~1 deg, comparable to typical accelerometer bias.")

    phi, th = df["phi_deg"].to_numpy(float), df["theta_deg"].to_numpy(float)
    out["gyro_vs_angle_rate"] = {}
    for tscale_name, tscale in (("t_ms=ms", 1e-3), ("t_ms=us", 1e-6), ("t_ms=s", 1.0)):
        sec = dt * tscale
        res: Dict[str, Any] = {}
        for ang_name, ang, g in (("phi", phi, "gx"), ("theta", th, "gy")):
            rate = np.diff(ang) / sec
            for k in (5, 15, 25, 50):
                rs = pd.Series(rate).rolling(k, min_periods=k).mean().to_numpy()
                gs = pd.Series(df[g].to_numpy(float)[1:]).rolling(k, min_periods=k).mean().to_numpy()
                res[f"d{ang_name}/dt~{g}@{k}"] = gain(gs, rs)
        out["gyro_vs_angle_rate"][tscale_name] = res
    out["gyro_note"] = ("gain ~ +1 => gyro is deg/s under that time hypothesis; ~ +57.3 => rad/s. The gain is bounded "
                        "by [ols, reverse] (errors-in-variables). Only the product (gyro unit x time unit) is constrained.")

    h = {}
    if {"tof_mm", "tof_h"} <= set(df.columns):
        h["tof_h~tof_mm"] = gain(df["tof_mm"].to_numpy(float), df["tof_h"].to_numpy(float))
    if {"z_est", "tof_h"} <= set(df.columns):
        h["z_est~tof_h"] = gain(df["tof_h"].to_numpy(float), df["z_est"].to_numpy(float))
    if {"z_set", "z_est"} <= set(df.columns):
        h["z_est~z_set"] = gain(df["z_set"].to_numpy(float), df["z_est"].to_numpy(float))
    out["height_channels"] = h
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--json")
    a = ap.parse_args()
    res = {Path(p).name: analyze(pd.read_csv(p)) for p in a.csv}
    text = json.dumps(res, indent=2)
    if a.json:
        Path(a.json).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
