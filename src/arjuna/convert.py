"""Convert CSV telemetry into an ARJUNA v1.1 ULog, so every upload goes through the same standard pipeline.

A known format (``config/arjuna/csv_formats.json``) maps every column exactly. Any other CSV is mapped by column
name, and each unit or frame decision is written into the conversion report. Values are converted to the standard's
SI units, radians and FLU body frame. Columns that cannot be mapped are dropped and listed. Missing values stay NaN.

The result is a real ``.ulg`` (with the hash chain) that ``load_flight`` reads like any ARJUNA log; the information
keys say it was converted from a CSV, which source format matched, and which fields were never recorded.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.arjuna.ulog import ArjunaFormatError, load_json
from src.arjuna.ulog_writer import Topic, ULogWriter

CFG = load_json("csv_formats.json")
MAX_ROWS = 1_000_000
MAX_COLUMNS = 300
G = 9.80665
ULOG_T0_US = 1_000_000
ARRAYS = {"gyro": 3, "accel": 3, "rates": 3, "accel_vibe": 3, "channels": 16, "motor_cmd": 8, "pwm_us": 8}
STANDARD_TOPICS = [
    "arjuna_vehicle_status", "arjuna_imu", "arjuna_imu_filtered", "arjuna_vibration", "arjuna_baro", "arjuna_gps",
    "arjuna_mag", "arjuna_rangefinder", "arjuna_optical_flow", "arjuna_vision_pose", "arjuna_attitude", "arjuna_position",
    "arjuna_estimator_innov", "arjuna_setpoints", "arjuna_rc_input", "arjuna_controller", "arjuna_actuator_outputs",
    "arjuna_esc_status", "arjuna_battery", "arjuna_fpga_timing", "arjuna_system_health", "arjuna_flight_event",
    "arjuna_mission_status", "arjuna_communication", "arjuna_firewall", "arjuna_fallback", "arjuna_proximity",
    "arjuna_detection",
]


class ConversionError(ArjunaFormatError):
    """The CSV cannot be converted (no time column, nothing recognisable, too large)."""


@dataclass
class Mapping:
    source: str
    targets: List[str]
    unit_in: str
    values: np.ndarray            # already in SI / radians / FLU
    note: str = ""


@dataclass
class Plan:
    fmt_id: str
    label: str
    basis: str
    time_col: str
    time_unit: str
    time_note: str
    mappings: List[Mapping] = field(default_factory=list)
    dropped: List[Dict[str, str]] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    min_airborne_s: float = 0.0
    min_alt_range_m: float = 2.0
    motor_layout: str = "unknown"
    rc_map: Dict[str, int] = field(default_factory=dict)
    air_rule: str = ""
    landed: Optional[np.ndarray] = None
    derived: List[Dict[str, str]] = field(default_factory=list)
    controller: Optional[Tuple[int, int]] = None   # (controller_id, axis) of the derived controller record


def _norm(h: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(h).strip().lower()).strip("_")


# ---- reading ---------------------------------------------------------------------------------------------------------

def read_table(raw: bytes) -> pd.DataFrame:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    head = text[:8192]
    try:
        sep = csv.Sniffer().sniff(head, delimiters=",;\t|").delimiter
    except csv.Error:
        sep = ","
    try:
        df = pd.read_csv(io.StringIO(text), sep=sep, low_memory=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ConversionError(f"The file could not be read as a table ({type(exc).__name__}).") from exc
    if df.shape[1] > MAX_COLUMNS:
        raise ConversionError(f"The file has {df.shape[1]} columns; the limit is {MAX_COLUMNS}.")
    if len(df) > MAX_ROWS:
        raise ConversionError(f"The file has {len(df):,} rows; the limit is {MAX_ROWS:,}.")
    if len(df) < 2:
        raise ConversionError("The file has fewer than two data rows.")
    return df


def _numeric(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    out, text_cols = {}, []
    for c in df.columns:
        v = pd.to_numeric(df[c], errors="coerce")
        if v.notna().mean() < 0.5 and df[c].notna().any():
            text_cols.append(str(c))
            continue
        out[str(c)] = v.astype(float)
    return pd.DataFrame(out), text_cols


# ---- units ------------------------------------------------------------------------------------------------------------

_TIME_SCALE = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}


def _time_unit_from_name(name: str) -> Optional[str]:
    n = _norm(name)
    for unit in ("us", "ms", "ns", "s"):
        if n.endswith("_" + unit) or n.endswith(unit) and n[: -len(unit)] in ("time", "timestamp", "t"):
            return unit
    return None


def _time_unit_from_steps(t: np.ndarray) -> str:
    d = np.diff(t[np.isfinite(t)])
    d = d[d > 0]
    step = float(np.median(d)) if len(d) else 1.0
    # plausible telemetry rates are 1-2000 Hz: step 0.0005-1 s
    for unit, scale in (("s", 1.0), ("ms", 1e-3), ("us", 1e-6), ("ns", 1e-9)):
        if 0.0004 <= step * scale <= 1.5:
            return unit
    return "s"


def _to_si(values: np.ndarray, unit: str) -> Tuple[np.ndarray, str]:
    v = values.astype(float)
    if unit == "deg":
        return np.radians(v), "degrees -> radians"
    if unit == "deg/s":
        return np.radians(v), "deg/s -> rad/s"
    if unit == "mm":
        return v / 1000.0, "mm -> m"
    if unit == "cm":
        return v / 100.0, "cm -> m"
    if unit == "g":
        return v * G, "g -> m/s^2"
    if unit == "mV":
        return v / 1000.0, "mV -> V"
    if unit == "flag":
        return np.where(np.isfinite(v), (v > 0).astype(float), np.nan), "non-zero -> true"
    return v, ""


# ---- planning: known format ---------------------------------------------------------------------------------------------

def _match_known(headers: List[str]) -> Optional[Tuple[Dict[str, Any], Dict[str, str]]]:
    by_norm = {_norm(h): h for h in headers}
    for fmt in CFG["formats"]:
        names = {f["from"] for f in fmt["fields"]} | set(fmt.get("dropped", {})) | {fmt["time"]["column"]}
        col: Dict[str, str] = {}
        for name in names:
            for cand in [name] + fmt.get("aliases", {}).get(name, []):
                if _norm(cand) in by_norm:
                    col[name] = by_norm[_norm(cand)]
                    break
        if all(r in col for r in fmt["required"]):
            return fmt, col
    return None


def _plan_known(fmt: Dict[str, Any], col: Dict[str, str], num: pd.DataFrame, t_s: np.ndarray) -> Plan:
    p = Plan(fmt["id"], fmt["label"], fmt.get("basis", ""), col[fmt["time"]["column"]], fmt["time"]["unit"],
             "unit from the format definition", min_airborne_s=float(fmt.get("min_airborne_s", 0.0)),
             motor_layout=fmt.get("motor_layout", "unknown"), rc_map=dict(fmt.get("rc_map", {})),
             min_alt_range_m=float(fmt.get("min_alt_range_m", 2.0)))
    p.assumptions = list(fmt.get("assumptions", []))
    if "min_airborne_basis" in fmt:
        p.assumptions.append(f"There is no minimum air time: any recording that gets airborne is analysed as a flight (a log that shows no "
                             f"movement and under {p.min_alt_range_m:g} m of height change is still treated as a bench test). {fmt['min_airborne_basis']}")
    for f in fmt["fields"]:
        h = col.get(f["from"])
        if h is None or h not in num:
            continue
        _add(p, h, f["to"], f.get("index"), f["unit"], num[h].to_numpy(float))
    used = {p.time_col} | {m.source for m in p.mappings}
    for name, why in fmt.get("dropped", {}).items():
        if name in col:
            p.dropped.append({"column": col[name], "reason": why})
            used.add(col[name])
    for h in num.columns:
        if h not in used:
            p.dropped.append({"column": h, "reason": "not part of this format's definition"})
    air = fmt.get("air", {})
    armed = num[col[air["armed"]]].to_numpy(float) > 0 if air.get("armed") in col else np.ones(len(num), bool)
    hcol = next((col[h] for h in air.get("height", []) if h in col), None)
    if hcol is not None:
        height = num[hcol].to_numpy(float)
        p.landed = np.where(armed & (height > air["height_threshold_m"]), 2, 0)
        p.air_rule = f"armed and {hcol} above {air['height_threshold_m']} m"
    return p


def _add(p: Plan, source: str, to: Any, index: Optional[int], unit: str, raw: np.ndarray) -> None:
    if to == "motor":
        if unit == "pwm_us":
            cmd, pwm, note = (raw - 1000.0) / 1000.0, raw, "PWM-like 1000-2000 -> motor_cmd 0-1"
        elif unit == "norm":
            cmd, pwm, note = raw, 1000.0 + 1000.0 * raw, "normalised 0-1 -> pwm 1000-2000"
        else:  # percent
            cmd, pwm, note = raw / 100.0, 1000.0 + 10.0 * raw, "percent -> motor_cmd 0-1"
        p.mappings.append(Mapping(source, [f"arjuna_actuator_outputs.motor_cmd[{index}]"], unit, np.clip(cmd, 0, 1.2), note))
        p.mappings.append(Mapping(source, [f"arjuna_actuator_outputs.pwm_us[{index}]"], unit, pwm, ""))
        return
    targets = to if isinstance(to, list) else [to]
    v, note = _to_si(raw, unit)
    p.mappings.append(Mapping(source, targets, unit, v, note))


# ---- planning: any other CSV --------------------------------------------------------------------------------------------

def _infer_unit(kind: str, name: str, v: np.ndarray, p: Plan) -> str:
    n = _norm(name)
    a = np.abs(v[np.isfinite(v)])
    big = float(np.percentile(a, 99)) if len(a) else 0.0
    if kind == "angle":
        if n.endswith("_deg") or "deg" in n:
            return "deg"
        if n.endswith("_rad") or "rad" in n:
            return "rad"
        unit = "deg" if big > 2 * math.pi + 0.5 else "rad"
        p.assumptions.append(f"'{name}': no unit in the name; values up to {big:.3g} imply {'degrees' if unit == 'deg' else 'radians'}.")
        return unit
    if kind == "rate":
        if "deg" in n or "dps" in n:
            return "deg/s"
        if "rad" in n:
            return "rad/s"
        unit = "deg/s" if big > 20 else "rad/s"
        p.assumptions.append(f"'{name}': no unit in the name; values up to {big:.3g} imply {unit}.")
        return unit
    if kind in ("accel", "accel_abs"):
        if n.endswith("_g"):
            return "g"
        med = float(np.median(a)) if len(a) else 0.0
        if kind == "accel" and 0.5 < med < 2.0 and big < 20:
            p.assumptions.append(f"'{name}': magnitudes near 1 imply g; converted to m/s^2.")
            return "g"
        return "m/s^2"
    if kind == "length":
        if n.endswith("_mm") or n.endswith("mm"):
            return "mm"
        if n.endswith("_cm") or n.endswith("cm"):
            return "cm"
        return "m"
    if kind == "volt":
        if big > 100:
            p.assumptions.append(f"'{name}': values above 100 imply millivolts.")
            return "mV"
        return "V"
    if kind == "motor":
        if big > 200:
            return "pwm_us"
        if big <= 1.05:
            return "norm"
        return "percent"
    return {"flag": "flag", "raw": "raw", "count": "count", "amp": "A"}.get(kind, "raw")


def _plan_generic(num: pd.DataFrame) -> Plan:
    g = CFG["generic"]
    by_norm = {_norm(h): h for h in num.columns}
    tcol = next((by_norm[n] for n in g["time"] if n in by_norm), None)
    if tcol is None:
        raise ConversionError("No time column was found. Name it time, timestamp, t, time_s, time_ms or time_us. "
                              f"Columns in the file: {', '.join(list(num.columns)[:40])}.")
    unit = _time_unit_from_name(tcol)
    note = "unit from the column name"
    if unit is None:
        unit = _time_unit_from_steps(num[tcol].to_numpy(float))
        note = f"no unit in the name; the median step implies {unit}"
    p = Plan("generic", "Unrecognised CSV, mapped by column name", "", tcol, unit, note,
             min_airborne_s=float(g["min_airborne_s"]), motor_layout=g["motor_layout"],
             min_alt_range_m=float(g["min_alt_range_m"]))
    used = {tcol}
    for spec in g["fields"]:
        if spec["to"] == "motor":
            for i in range(8):
                h = next((by_norm[n.format(n=i + 1)] for n in spec["names"] if n.format(n=i + 1) in by_norm), None)
                if h and h not in used:
                    v = num[h].to_numpy(float)
                    _add(p, h, "motor", i, _infer_unit("motor", h, v, p), v)
                    used.add(h)
            continue
        h = next((by_norm[n] for n in spec["names"] if n in by_norm and by_norm[n] not in used), None)
        if h is None:
            continue
        v = num[h].to_numpy(float)
        _add(p, h, spec["to"], None, _infer_unit(spec["kind"], h, v, p), v)
        used.add(h)
    for h in num.columns:
        if h not in used:
            p.dropped.append({"column": h, "reason": "name not recognised as an ARJUNA signal"})
    _frame_check(p)
    return p


def _frame_check(p: Plan) -> None:
    """A body z accelerometer that reads about -g at rest is FRD (PX4 / ArduPilot); ARJUNA is FLU: flip y and z."""
    az = next((m for m in p.mappings if "arjuna_imu.accel[2]" in m.targets), None)
    if az is None:
        return
    v = az.values[np.isfinite(az.values)]
    if len(v) and np.median(v) < -5.0:
        flip = {"arjuna_imu.accel[1]", "arjuna_imu.accel[2]", "arjuna_imu.gyro[1]", "arjuna_imu.gyro[2]",
                "arjuna_attitude.rates[1]", "arjuna_attitude.rates[2]", "arjuna_attitude.pitch", "arjuna_attitude.yaw"}
        for m in p.mappings:
            if set(m.targets) & flip:
                m.values = -m.values
                m.note = (m.note + "; " if m.note else "") + "sign flipped FRD -> FLU"
        p.assumptions.append("The z accelerometer reads about -9.8 m/s^2 at rest, so the CSV uses the FRD body frame "
                             "(PX4 / ArduPilot). y and z axes, pitch and yaw were negated to convert to ARJUNA FLU.")


def _generic_air(p: Plan, n: int) -> None:
    if p.landed is not None:
        return
    g = CFG["generic"]["air"]
    get = lambda target: next((m.values for m in p.mappings if target in m.targets), None)  # noqa: E731
    armed = get("arjuna_vehicle_status.armed")
    armed_b = armed > 0 if armed is not None else np.ones(n, bool)
    height = next((v for v in (get("arjuna_rangefinder.distance"), get("arjuna_position.pos_u"), get("arjuna_baro.altitude"))
                   if v is not None), None)
    cmds = [m.values for m in p.mappings if m.targets[0].startswith("arjuna_actuator_outputs.motor_cmd")]
    if height is not None:
        base = float(np.nanmedian(height[: max(5, n // 50)]))
        p.landed = np.where(armed_b & (height - base > g["height_threshold_m"]), 2, 0)
        p.air_rule = f"armed and height more than {g['height_threshold_m']} m above its starting value"
    elif cmds:
        mean = np.nanmean(np.vstack(cmds), axis=0)
        p.landed = np.where(armed_b & (mean > g["motor_threshold"]), 2, 0)
        p.air_rule = f"armed and mean motor command above {g['motor_threshold']}"
    else:
        p.landed = np.full(n, 2)
        p.air_rule = "no arming, height or motor columns: the whole recording is treated as in the air"
    p.assumptions.append(f"In the air = {p.air_rule} (the CSV has no landed-state column).")


# ---- derived standard records ------------------------------------------------------------------------------------------

VIBE_HP_HZ = 5.0     # like ArduPilot VIBE: accel minus its 5 Hz low-pass, RMS smoothed at 2 Hz
VIBE_LP_HZ = 2.0


def _lowpass(v: np.ndarray, fc: float, dt: float) -> np.ndarray:
    a = dt / (1.0 / (2 * math.pi * fc) + dt)
    out = np.empty_like(v)
    acc = v[np.isfinite(v)][0] if np.isfinite(v).any() else 0.0
    for i, x in enumerate(v):
        if np.isfinite(x):
            acc += a * (x - acc)
        out[i] = acc
    return out


def _derive(p: Plan, t_s: np.ndarray) -> None:
    get = lambda target: next((m.values for m in p.mappings if target in m.targets), None)  # noqa: E731
    d = np.diff(t_s)
    dt = float(np.median(d[d > 0])) if (d > 0).any() else 0.0
    have_vibe = any(t.startswith("arjuna_vibration.accel_vibe") for m in p.mappings for t in m.targets)
    if not have_vibe and dt and 1.0 / dt >= 4 * VIBE_HP_HZ:
        made = False
        for i in range(3):
            acc = get(f"arjuna_imu.accel[{i}]")
            if acc is None:
                continue
            hp = acc - _lowpass(acc, VIBE_HP_HZ, dt)
            vibe = np.sqrt(_lowpass(np.where(np.isfinite(hp), hp ** 2, np.nan), VIBE_LP_HZ, dt))
            p.mappings.append(Mapping(f"derived from arjuna_imu.accel[{i}]", [f"arjuna_vibration.accel_vibe[{i}]"], "m/s^2", vibe,
                                      "computed by the converter"))
            made = True
        if made:
            p.derived.append({"record": "arjuna_vibration.accel_vibe[0..2]",
                              "how": f"accelerometer minus its {VIBE_HP_HZ:g} Hz low-pass, RMS smoothed at {VIBE_LP_HZ:g} Hz "
                                     "(the ArduPilot VIBE method)",
                              "caveat": f"at {1 / dt:.0f} Hz logging this misses vibration above {0.5 / dt:.0f} Hz, so it "
                                        "underestimates what the on-board ARJUNA value would show"})
    sp, meas = get("arjuna_setpoints.alt_sp"), get("arjuna_position.pos_u")
    if sp is not None and meas is not None:
        p.mappings.append(Mapping("derived from alt_sp and pos_u", ["arjuna_controller.setpoint"], "m", sp, "computed by the converter"))
        p.mappings.append(Mapping("derived from alt_sp and pos_u", ["arjuna_controller.measured"], "m", meas, ""))
        p.mappings.append(Mapping("derived from alt_sp and pos_u", ["arjuna_controller.error"], "m", sp - meas, ""))
        p.controller = (2, 5)  # altitude loop, z axis
        p.derived.append({"record": "arjuna_controller (altitude, z)",
                          "how": "setpoint = altitude setpoint, measured = altitude estimate, error = setpoint - measured",
                          "caveat": "P, I, D terms and saturation are not in the CSV"})


# ---- writing ----------------------------------------------------------------------------------------------------------

def _topics(p: Plan) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {"arjuna_vehicle_status": {}}
    for m in p.mappings:
        for t in m.targets:
            topic, fld = t.split(".", 1)
            base = fld.split("[", 1)[0]
            out.setdefault(topic, {})[base] = ARRAYS.get(base, 1)
    return out


def _write(p: Plan, t_us: np.ndarray, info: Dict[str, str]) -> Tuple[bytes, Dict[str, List[str]]]:
    spec = _topics(p)
    defs: Dict[str, Topic] = {}
    for topic, fields in spec.items():
        cols = [("uint64_t", "timestamp", 1)]
        if topic == "arjuna_vehicle_status":
            cols += [("uint8_t", "flight_mode", 1), ("bool", "armed", 1), ("uint8_t", "landed_state", 1),
                     ("bool", "failsafe_active", 1)]
        elif topic == "arjuna_actuator_outputs":
            cols.append(("uint8_t", "motor_count", 1))
        elif topic == "arjuna_controller":
            cols += [("uint8_t", "controller_id", 1), ("uint8_t", "axis", 1)]
        cols += [("float", base, size) for base, size in fields.items() if not (topic == "arjuna_vehicle_status" and base == "armed")]
        defs[topic] = Topic(topic, cols)
    n = len(t_us)
    cols_by_topic: Dict[str, Dict[str, np.ndarray]] = {t: {} for t in spec}
    for m in p.mappings:
        for t in m.targets:
            topic, fld = t.split(".", 1)
            cols_by_topic[topic][fld] = m.values
    armed = cols_by_topic["arjuna_vehicle_status"].pop("armed", None)
    n_motors = sum(1 for k in cols_by_topic.get("arjuna_actuator_outputs", {}) if k.startswith("motor_cmd"))
    fd, path = tempfile.mkstemp(suffix=".ulg")
    os.close(fd)
    try:
        w = ULogWriter(path, int(t_us[0]))
        for k, v in info.items():
            w.info(k, v)
        for tp in defs.values():
            w.add_topic(tp)
        w.write_formats()
        for role, ch in p.rc_map.items():
            w.param(f"RCMAP_{role}", int(ch))
        if n_motors:
            w.param("MOT_PWM_MAX", 2000.0)
        w.begin_data()
        w.log(6, int(t_us[0]), f"Converted from CSV ({p.label}) by {CFG['converter_version']}")
        landed = p.landed if p.landed is not None else np.full(n, 255)
        for i in range(n):
            ts = int(t_us[i])
            for topic, tp in defs.items():
                vals: Dict[str, Any] = {"timestamp": ts}
                if topic == "arjuna_vehicle_status":
                    a = armed[i] if armed is not None else 1.0
                    vals.update(flight_mode=255, armed=bool(a > 0) if np.isfinite(a) else False,
                                landed_state=int(landed[i]), failsafe_active=False)
                elif topic == "arjuna_actuator_outputs":
                    vals["motor_count"] = n_motors
                elif topic == "arjuna_controller" and p.controller:
                    vals["controller_id"], vals["axis"] = p.controller
                cols = cols_by_topic[topic]
                for _, base, size in tp.fields[1:]:
                    if base in vals:
                        continue
                    if size > 1:
                        vals[base] = [float(cols[f"{base}[{k}]"][i]) if f"{base}[{k}]" in cols else math.nan for k in range(size)]
                    else:
                        vals[base] = float(cols[base][i]) if base in cols else math.nan
                w.data(topic, 0, vals)
        w.close()
        data = open(path, "rb").read()
    finally:
        if os.path.exists(path):
            os.unlink(path)
    filled = {t: sorted(c) for t, c in cols_by_topic.items()}
    return data, filled


# ---- entry point ------------------------------------------------------------------------------------------------------

def convert_csv(raw: bytes, filename: str) -> Tuple[bytes, Dict[str, Any]]:
    """Return (ULog bytes, conversion report)."""
    df = read_table(raw)
    num, text_cols = _numeric(df)
    known = _match_known(list(num.columns))
    if known:
        fmt, col = known
        tcol = col[fmt["time"]["column"]]
    else:
        fmt = None
    p0 = _plan_generic(num) if fmt is None else None
    tcol = p0.time_col if p0 else tcol
    unit = p0.time_unit if p0 else fmt["time"]["unit"]
    t_raw = num[tcol].to_numpy(float)
    ok = np.isfinite(t_raw)
    order = np.argsort(t_raw[ok], kind="stable")
    idx = np.where(ok)[0][order]
    t_sorted = t_raw[idx]
    keep = np.concatenate([[True], np.diff(t_sorted) > 0])  # first sample of each timestamp
    idx = idx[keep]
    n_dup = int((~keep).sum())
    if len(idx) < 2:
        raise ConversionError(f"Column '{tcol}' has fewer than two valid timestamps.")
    num = num.iloc[idx].reset_index(drop=True)
    t_s = (num[tcol].to_numpy(float) - num[tcol].iloc[0]) * _TIME_SCALE[unit]
    p = _plan_known(fmt, col, num, t_s) if fmt else _plan_generic(num)
    _generic_air(p, len(num))
    _derive(p, t_s)
    if not [m for m in p.mappings if not m.targets[0].startswith("arjuna_vehicle_status")]:
        raise ConversionError("No column could be mapped to an ARJUNA signal. Recognised names include roll, pitch, "
                              "gx/gyro_x, ax/acc_x, m1..m4/motor1, voltage, alt.")
    t_us = (ULOG_T0_US + t_s * 1e6).astype(np.int64)
    d = np.diff(t_s)
    rate = 1.0 / float(np.median(d[d > 0])) if (d > 0).any() else None
    sha = hashlib.sha256(raw).hexdigest()
    spec = _topics(p)
    produced = set(spec)
    info = {
        "sys_name": "ARJUNA", "flight_id": f"CSV-{sha[:6].upper()}", "vehicle_id": "unknown",
        "schema_version": "1.1.0", "log_profile": "standard", "frame_nav": "ENU", "frame_body": "FLU",
        "source_format": "csv", "source_profile": p.fmt_id, "source_file": filename[:120], "source_sha256": sha,
        "converter_version": CFG["converter_version"], "min_airborne_s": f"{p.min_airborne_s:g}",
        "min_alt_range_m": f"{p.min_alt_range_m:g}",
        "motor_layout": p.motor_layout,
        "topics_not_produced": ",".join(t for t in STANDARD_TOPICS if t not in produced),
    }
    data, filled = _write(p, t_us, info)
    report = {
        "converter_version": CFG["converter_version"], "source_file": filename, "source_sha256": sha,
        "format": {"id": p.fmt_id, "label": p.label, "basis": p.basis},
        "time": {"column": p.time_col, "unit": unit, "how": p.time_note,
                 "duration_s": round(float(t_s[-1]), 3), "sample_rate_hz": round(rate, 2) if rate else None},
        "rows_in": int(len(df)), "rows_written": int(len(num)), "rows_without_valid_time": int((~ok).sum()),
        "duplicate_timestamps_dropped": n_dup,
        "mapped": [{"from": m.source, "to": m.targets, "unit_in": m.unit_in, "conversion": m.note} for m in p.mappings
                   if not m.source.startswith("derived")],
        "derived": p.derived,
        "dropped": p.dropped + [{"column": c, "reason": "text, not numeric"} for c in text_cols],
        "assumptions": p.assumptions, "air_rule": p.air_rule, "min_airborne_s": p.min_airborne_s,
        "motor_layout": p.motor_layout, "topics_written": sorted(produced), "ulog_bytes": len(data),
    }
    return data, report
