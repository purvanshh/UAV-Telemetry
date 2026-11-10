"""Measure the TRAINING data's sensor conventions (rather than assuming them).

Reads ArduPilot-extracted IMU/ATT CSVs (e.g. ``~/Downloads/LOGS 2/uav dataset``) and reports, from the
quiet start of each log: AccZ sign, AccY vs -g*sin(Roll), AccX vs +g*sin(Pitch) and the per-log AccX/AccY
bias scatter, plus gyro-vs-attitude-rate gain (expected +57.3 deg/s per rad/s).

    python scripts/check_training_convention.py "/path/to/uav dataset"
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

G = 9.80665


def gain(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    b_yx = np.polyfit(x, y, 1)[0]
    b_xy = np.polyfit(y, x, 1)[0]
    return float(b_yx), float(1 / b_xy)


def main(base: str) -> dict:
    rows, g_roll, g_pitch = [], [], []
    for f in sorted(glob.glob(os.path.join(base, "IMU", "*_IMU.csv"))):
        n = os.path.basename(f)[:8]
        try:
            imu = pd.read_csv(f)
            att = pd.read_csv(os.path.join(base, "ATT", f"{n}_ATT.csv"))
        except Exception:
            continue
        if "I" in imu:
            imu = imu[imu.I == 0]
        t = imu.TimeUS.to_numpy(float)
        s = imu[(t - t[0]) < 4e6]
        if len(s) >= 100:
            gq = np.abs(s[["GyrX", "GyrY", "GyrZ"]]).max().max()
            an = np.linalg.norm(s[["AccX", "AccY", "AccZ"]], axis=1).mean()
            if gq < 0.05 and abs(an - G) < 0.6:
                rows.append({"log": n, "roll": float(np.interp(s.TimeUS, att.TimeUS, att.Roll).mean()),
                             "pitch": float(np.interp(s.TimeUS, att.TimeUS, att.Pitch).mean()),
                             "ax": float(s.AccX.mean()), "ay": float(s.AccY.mean()), "az": float(s.AccZ.mean())})
        if len(att) >= 400:
            ta = att.TimeUS.to_numpy(float) / 1e6
            dt = np.diff(ta)
            ok = np.where(dt < 0.5, dt, np.nan)
            for ang, gyr, store in (("Roll", "GyrX", g_roll), ("Pitch", "GyrY", g_pitch)):
                gg = np.interp(att.TimeUS, imu.TimeUS, imu[gyr])
                rate = pd.Series(np.diff(att[ang].to_numpy()) / ok).rolling(10, min_periods=10).mean().to_numpy()
                gs = pd.Series(gg[1:]).rolling(10, min_periods=10).mean().to_numpy()
                store.append(gain(gs, rate))
    R = pd.DataFrame(rows)
    r, p = np.deg2rad(R.roll), np.deg2rad(R.pitch)
    pred = {"AccX": G * np.sin(p), "AccY": -G * np.sin(r) * np.cos(p), "AccZ": -G * np.cos(r) * np.cos(p)}
    out = {"quiet_logs": int(len(R)), "AccZ_mean": float(R.az.mean()),
           "AccX_bias_mean_std": [float((R.ax - pred["AccX"]).mean()), float((R.ax - pred["AccX"]).std())],
           "AccY_bias_mean_std": [float((R.ay - pred["AccY"]).mean()), float((R.ay - pred["AccY"]).std())],
           "AccY_corr_with_minus_g_sin_roll": float(np.corrcoef(pred["AccY"], R.ay)[0, 1]),
           "AccX_corr_with_g_sin_pitch": float(np.corrcoef(pred["AccX"], R.ax)[0, 1]),
           "pitch_range_deg": [float(R.pitch.min()), float(R.pitch.max())],
           "roll_range_deg": [float(R.roll.min()), float(R.roll.max())],
           "gyro_gain_roll_median(ols,reverse)": np.median(np.array(g_roll), axis=0).tolist(),
           "gyro_gain_pitch_median(ols,reverse)": np.median(np.array(g_pitch), axis=0).tolist()}
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Downloads/LOGS 2/uav dataset"))
