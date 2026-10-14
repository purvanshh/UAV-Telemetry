
"""UAV preprocessing v3.0 — session-aware alignment and windowing."""
from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd

def _norm(s): return re.sub(r"[\s_]+", "", str(s)).lower()

def resolve(cols, name):
    lut = {_norm(c): c for c in cols}
    return lut.get(_norm(name))

DEFAULT_SCHEMA = {
    "IMU": {"ts_priority":["TimeUS","Timestamp"], "ts_unit":"us",
            "features":{"gyro_x":"GyrX","gyro_y":"GyrY","gyro_z":"GyrZ",
                        "acc_x":"AccX","acc_y":"AccY","acc_z":"AccZ"}},
    "ATT": {"ts_priority":["TimeUS","Timestamp"], "ts_unit":"us",
            "features":{"roll":"Roll","pitch":"Pitch","yaw":"Yaw",
                        "des_roll":"DesRoll","des_pitch":"DesPitch","des_yaw":"DesYaw"}},
    "BARO":{"ts_priority":["TimeUS","Timestamp"], "ts_unit":"us",
            "features":{"alt":"Alt","alt_amsl":"AltAMSL","press":"Press","temp":"Temp"}},
}

UNIT_MULT = {"us":1e6, "ms":1e3, "s":1.0, "ns":1e9}

DEFAULT_ALIGN = {"target_hz":10.0, "max_gap_s":0.5, "session_gap_s":1.0,
                 "features":["gyro_x","gyro_y","gyro_z","acc_x","acc_y","acc_z",
                             "roll","pitch","yaw_sin","yaw_cos",
                             "roll_res","pitch_res",
                             "alt","press","temp","alt_rate","press_rate"]}

def load_stream(path, spec):
    df = pd.read_csv(path)
    return load_stream_df(df, spec)


def load_stream_df(df, spec):
    """Same as load_stream but accepts an in-memory DataFrame (upload path)."""
    ts_col = None
    for c in spec["ts_priority"]:
        r = resolve(list(df.columns), c)
        if r is not None:
            ts_col = r
            break
    if ts_col is None:
        raise ValueError("no timestamp column")
    t_raw = pd.to_numeric(df[ts_col], errors="coerce").astype("float64")
    unit = spec.get("ts_unit", "us")
    out = df.copy()
    out["__t_s"] = (t_raw / UNIT_MULT[unit]).values
    out = (
        out.dropna(subset=["__t_s"])
        .sort_values("__t_s")
        .drop_duplicates("__t_s")
        .reset_index(drop=True)
    )
    return out

def gap_aware_interp(t_src, y_src, t_grid, max_gap):
    t_src = np.asarray(t_src, dtype=float)
    y_src = np.asarray(y_src, dtype=float)
    t_grid = np.asarray(t_grid, dtype=float)
    out = np.interp(t_grid, t_src, y_src, left=np.nan, right=np.nan)
    if len(t_src) < 2: return out
    li = np.clip(np.searchsorted(t_src, t_grid, side="right") - 1, 0, len(t_src) - 1)
    ri = np.clip(np.searchsorted(t_src, t_grid, side="left"),       0, len(t_src) - 1)
    interval = np.abs(t_src[ri] - t_src[li])
    exact = (t_src[li] == t_grid) | (t_src[ri] == t_grid)
    out[(interval > max_gap) & ~exact] = np.nan
    return out

def safe_diff(y, t, max_gap):
    d = np.diff(y, prepend=np.nan); dt = np.diff(t, prepend=np.nan)
    d[dt > max_gap] = np.nan
    return d

def compute_session_id(grid, source_ts_list, session_gap_s):
    n = len(grid)
    in_gap = np.zeros(n, dtype=bool)
    for ts in source_ts_list:
        if len(ts) < 2: continue
        dt = np.diff(ts)
        for i in np.where(dt > session_gap_s)[0]:
            in_gap |= (grid > ts[i]) & (grid < ts[i+1])
    ids = np.full(n, -1, dtype=np.int32)
    sid = -1
    prev_gap = True
    for i in range(n):
        if not in_gap[i]:
            if prev_gap: sid += 1
            ids[i] = sid
            prev_gap = False
        else:
            prev_gap = True
    return ids

def align_flight(imu, att, baro, schema=DEFAULT_SCHEMA, cfg=DEFAULT_ALIGN):
    t_lo = max(imu["__t_s"].iloc[0], att["__t_s"].iloc[0], baro["__t_s"].iloc[0])
    t_hi = min(imu["__t_s"].iloc[-1], att["__t_s"].iloc[-1], baro["__t_s"].iloc[-1])
    if t_hi <= t_lo: raise ValueError("no temporal overlap")
    grid = np.arange(t_lo, t_hi, 1.0/cfg["target_hz"])
    mg = cfg["max_gap_s"]

    def col(df, name):
        r = resolve(list(df.columns), name)
        return df[r].astype(float).values if r is not None else None

    out = pd.DataFrame({"_t_s": grid})
    for k, c in schema["IMU"]["features"].items():
        v = col(imu, c)
        out[k] = gap_aware_interp(imu["__t_s"].values, v, grid, mg) if v is not None else np.nan
    for k in ("roll","pitch","yaw","des_roll","des_pitch"):
        c = schema["ATT"]["features"][k]; v = col(att, c)
        out[k] = gap_aware_interp(att["__t_s"].values, v, grid, mg) if v is not None else np.nan
    yr = np.deg2rad(out["yaw"].values)
    out["yaw_sin"] = np.sin(yr); out["yaw_cos"] = np.cos(yr)
    out = out.drop(columns=["yaw"])
    out["roll_res"]  = out["roll"]  - out["des_roll"]
    out["pitch_res"] = out["pitch"] - out["des_pitch"]
    out = out.drop(columns=["des_roll","des_pitch"])
    for k, c in schema["BARO"]["features"].items():
        if k == "alt_amsl": continue
        v = col(baro, c)
        out[k] = gap_aware_interp(baro["__t_s"].values, v, grid, mg) if v is not None else np.nan
    out["alt_rate"]   = safe_diff(out["alt"].values,   out["_t_s"].values, mg)
    out["press_rate"] = safe_diff(out["press"].values, out["_t_s"].values, mg)

    sid = compute_session_id(grid,
                             [imu["__t_s"].values, att["__t_s"].values, baro["__t_s"].values],
                             cfg["session_gap_s"])
    out["_session_id"] = sid

    miss = [f for f in cfg["features"] if f not in out.columns]
    if miss: raise ValueError(f"missing: {miss}")
    feats = out[cfg["features"]].astype(float).replace([np.inf,-np.inf], np.nan)
    return pd.concat([out[["_t_s","_session_id"]], feats], axis=1)

def make_windows(df, features, win_len=30, stride=15, session_gap_s=1.0):
    """Session-aware windows. Never crosses a source-data gap.

    session_gap_s is accepted for signature compatibility; the actual
    segmentation was applied inside align_flight via _session_id.
    """
    if "_session_id" not in df.columns:
        raise ValueError("align_flight must be called before make_windows; "
                         "df is missing _session_id")
    t   = df["_t_s"].values
    sid = df["_session_id"].values
    Xs, meta = [], []
    for s in np.unique(sid):
        if s < 0: continue
        idx = np.where(sid == s)[0]
        if len(idx) < win_len: continue
        sub_X = df[features].values[idx].astype(np.float32)
        sub_t = t[idx]
        for s0 in range(0, len(sub_X) - win_len + 1, stride):
            w = sub_X[s0:s0+win_len]
            if np.isnan(w).any(): continue
            Xs.append(w)
            meta.append({"session_id": int(s),
                         "t_start": float(sub_t[s0]),
                         "t_end":   float(sub_t[s0+win_len-1])})
    if not Xs:
        return np.empty((0, win_len, len(features)), np.float32), meta
    return np.stack(Xs), meta