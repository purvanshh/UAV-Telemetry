"""Layer 1 of ARJUNA fault detection: rule checks for every fault code (Logging Standard v1.1, p. 31-32).

Thresholds come from ``config/arjuna/thresholds.json`` (approved by Labhesh, 16 Nov 2026); the logic follows the
approved triage (tools/triage.py) so findings agree with the dataset labels. A few checks implement fault
definitions from the guide that have no approved threshold yet; they say so and carry ``confidence: "medium"``.

Every finding has a fault code, severity, start and end time (seconds since log start), the detector that found it
and its evidence: topic, field, instance, statistic and value. Codes that cannot be evaluated on this log are
reported with the reason, never silently skipped.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.arjuna.ulog import ArjunaFlight, enum_name, load_json

TH = load_json("thresholds.json")

FAULT_CODES: Dict[str, str] = {
    "VIB_HIGH": "Excessive vibration or IMU clipping",
    "SENSOR_FAIL": "A sensor stops, freezes or reports nonsense",
    "GPS_DEGRADED": "GPS loss, jumps or poor accuracy",
    "EST_DIVERGE": "Estimator inconsistent or reset",
    "CTRL_OSC": "Oscillation, poor tuning",
    "CTRL_SAT": "Controller or motors at their limit",
    "MOTOR_ESC": "Motor or ESC fault, imbalance, damaged propeller",
    "POWER": "Battery sag, brownout, rail voltage dip",
    "TIMING": "Deadline misses, jitter, stale data",
    "COMMS": "Link loss, heartbeat timeout, packet loss",
    "SECURITY": "Rejected or suspicious commands",
    "LOGGING": "Dropouts in the log itself",
    "CRASH": "Impact or uncommanded descent",
}

# Provisional event code -> (fault code, severity), as in the approved triage.
EVENT_FLAGS = {
    23: ("CRASH", "critical"), 33: ("MOTOR_ESC", "critical"), 24: ("VIB_HIGH", "critical"),
    7: ("GPS_DEGRADED", "critical"), 8: ("GPS_DEGRADED", "warning"), 11: ("COMMS", "critical"),
    12: ("COMMS", "critical"), 13: ("POWER", "critical"), 10: ("POWER", "warning"), 14: ("EST_DIVERGE", "critical"),
    17: ("EST_DIVERGE", "warning"), 18: ("EST_DIVERGE", "warning"), 25: ("SENSOR_FAIL", "warning"),
    29: ("SENSOR_FAIL", "warning"), 36: ("SENSOR_FAIL", "warning"), 20: ("CRASH", "warning"),
    35: ("GPS_DEGRADED", "critical"), 32: ("EST_DIVERGE", "warning"), 31: ("GPS_DEGRADED", "warning"),
}
FAILSAFE_CODE = {"low_battery": "POWER", "rc_link_loss": "COMMS", "gcs_link_loss": "COMMS", "gps_loss": "GPS_DEGRADED",
                 "ekf_navigation": "EST_DIVERGE", "vibration": "VIB_HIGH", "sensors": "SENSOR_FAIL",
                 "dead_reckoning": "GPS_DEGRADED"}

# Guide definitions without an approved threshold yet (confidence "medium").
FREEZE_MIN_S = 2.0          # a continuous sensor channel exactly constant this long in the air
UPDATE_GAP_FACTOR = 5.0     # a gap this many times the topic's median period, in the air
RATE_WINDOW_S = 2.0         # rolling window for tracking-error RMS
MERGE_GAP_S = 2.0           # exceedances closer than this are one interval
EVENT_PAD_S = 5.0           # an instantaneous event is shown as +-5 s, like the dataset labels

R2D = 180.0 / np.pi


def _r(x: Any, nd: int = 3) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def clock(t: Optional[float]) -> str:
    if t is None:
        return "?"
    s = max(0.0, float(t))
    return f"{int(s // 60)}:{s % 60:04.1f}"


def intervals(t: np.ndarray, mask: np.ndarray, merge_gap: float = MERGE_GAP_S, min_len: float = 0.0) -> List[Tuple[float, float]]:
    """Contiguous runs of ``mask`` as (start, end) times, merging runs separated by less than ``merge_gap``."""
    out: List[List[float]] = []
    idx = np.where(mask)[0]
    if not len(idx):
        return []
    start = prev = idx[0]
    for i in idx[1:]:
        if t[i] - t[prev] > merge_gap:
            out.append([t[start], t[prev]])
            start = i
        prev = i
    out.append([t[start], t[prev]])
    return [(float(a), float(b)) for a, b in out if b - a >= min_len]


class Findings:
    def __init__(self, flight: ArjunaFlight):
        self.flight = flight
        self.items: List[Dict[str, Any]] = []
        self.coverage: Dict[str, Dict[str, Any]] = {c: {"status": "not_evaluable", "checks": [], "reasons": []}
                                                    for c in FAULT_CODES}
        self.events_only: set = set()

    def ran(self, code: str, check: str) -> None:
        cov = self.coverage[code]
        cov["status"] = "evaluated"
        if check not in cov["checks"]:
            cov["checks"].append(check)

    def cannot(self, code: str, reason: str) -> None:
        if reason not in self.coverage[code]["reasons"]:
            self.coverage[code]["reasons"].append(reason)

    def add(self, code: str, severity: str, title: str, t_start: float, t_end: float, detector: str,
            evidence: List[Dict[str, Any]], *, confidence: str = "high", basis: str = "", window: str = "interval") -> None:
        self.items.append({"code": code, "severity": severity, "title": title, "t_start": _r(t_start, 2),
                           "t_end": _r(t_end, 2), "when": f"{clock(t_start)}–{clock(t_end)}", "window": window,
                           "detector": detector, "confidence": confidence, "basis": basis, "evidence": evidence})


def ev(topic: str, field: str, stat: str, value: Any, unit: Optional[str] = None, t: Optional[float] = None,
       instance: int = 0, **extra: Any) -> Dict[str, Any]:
    out = {"topic": topic, "field": field, "instance": instance, "stat": stat, "value": _r(value, 4), "unit": unit}
    if t is not None:
        out["t"] = _r(t, 2)
        out["at"] = clock(t)
    out.update(extra)
    return out


def _pad(t: float, a0: float, a1: float) -> Tuple[float, float]:
    return max(a0, t - EVENT_PAD_S), min(max(a1, t), t + EVENT_PAD_S)


# ---- individual checks ---------------------------------------------------------------------------------------------

def check_events(fl: ArjunaFlight, F: Findings, a0: float, a1: float, t_takeoff: Optional[float]) -> None:
    fe = fl.get("arjuna_flight_event")
    if fe is None:
        for c in ("CRASH", "COMMS"):
            F.cannot(c, "no arjuna_flight_event topic")
        return
    for c in {v[0] for v in EVENT_FLAGS.values()}:
        F.events_only.add(c)  # events alone are a partial check: they only show what the autopilot itself raised
    yaw_resets = []
    t = fe["t"].to_numpy(float)
    codes = fe["event_code"].to_numpy(float)
    for ti, ci in zip(t, codes):
        if not np.isfinite(ci) or int(ci) not in EVENT_FLAGS:
            continue
        c = int(ci)
        code, sev = EVENT_FLAGS[c]
        name = enum_name("EVENT_CODES", c)
        if c in (17, 18, 32) and (not fl.in_air(np.array([ti]))[0] or t_takeoff is None or ti - t_takeoff < 60):
            continue  # estimator housekeeping on the ground or right after take-off is normal
        if c == 17:
            yaw_resets.append(ti)
            continue
        s, e = _pad(ti, a0, a1) if a1 > a0 else (ti, ti)
        F.add(code, sev, f"Flight event: {name.replace('_', ' ')}", s, e, "event_log",
              [ev("arjuna_flight_event", "event_code", "event", c, None, ti, name=name)],
              basis="Event raised by the flight controller.")
    if len(yaw_resets) >= 2:
        F.add("EST_DIVERGE", "warning", f"{len(yaw_resets)} estimator yaw resets in the air", yaw_resets[0], yaw_resets[-1],
              "event_log", [ev("arjuna_flight_event", "event_code", "count of ekf_yaw_reset", len(yaw_resets), "count", yaw_resets[0])],
              basis="Two or more yaw resets in the air.")


def check_failsafe(fl: ArjunaFlight, F: Findings) -> None:
    vs = fl.get("arjuna_vehicle_status")
    if vs is None or "failsafe_active" not in vs:
        return
    t = vs["t"].to_numpy(float)
    act = vs["failsafe_active"].to_numpy(float) > 0
    for s, e in intervals(t, act, merge_gap=1.0):
        sel = (t >= s) & (t <= e)
        reasons = vs.loc[sel, "failsafe_reason"].dropna().unique() if "failsafe_reason" in vs else []
        names = [enum_name("FAILSAFE_REASON", r) for r in reasons]
        code = next((FAILSAFE_CODE[n] for n in names if n in FAILSAFE_CODE), None)
        if code is None:
            continue
        F.add(code, "warning", f"Failsafe active ({', '.join(n.replace('_', ' ') for n in names)})", s, e, "rule:failsafe",
              [ev("arjuna_vehicle_status", "failsafe_active", "duration", e - s, "s", s, reasons=names)],
              basis="The vehicle reported an active failsafe.")


def check_vibration(fl: ArjunaFlight, F: Findings) -> None:
    if not fl.has("arjuna_vibration"):
        F.cannot("VIB_HIGH", "no arjuna_vibration topic")
        return
    for inst in fl.instances("arjuna_vibration"):
        vb = fl.get("arjuna_vibration", inst)
        t = vb["t"].to_numpy(float)
        m = fl.in_air(t)
        if not m.any():
            continue
        axes = [f"accel_vibe[{i}]" for i in range(3) if f"accel_vibe[{i}]" in vb]
        if axes:
            F.ran("VIB_HIGH", "accel vibration level")
            vib = np.vstack([vb[a].to_numpy(float) for a in axes])
            vib_air = np.where(m, vib, np.nan)
            mx = np.nanmax(vib_air, axis=1)
            p95 = np.nanpercentile(vib_air, 95, axis=1)
            ax = int(np.nanargmax(mx))
            peak_i = int(np.nanargmax(vib_air[ax]))
            worst = np.nanmax(vib_air, axis=0)
            if mx[ax] > TH["vibe_crit"]:
                runs = intervals(t, worst > TH["vibe_crit"])
                s, e = next(((a, b) for a, b in runs if a <= t[peak_i] <= b), (t[peak_i], t[peak_i]))
                F.add("VIB_HIGH", "critical", f"Vibration {'xyz'[ax]} axis up to {mx[ax]:.0f} m/s²", s, e, "rule:vibe_crit",
                      [ev("arjuna_vibration", axes[ax], "max in air", mx[ax], "m/s^2", t[peak_i], inst),
                       ev("arjuna_vibration", axes[ax], "p95 in air", p95[ax], "m/s^2", None, inst)],
                      basis=f"Peak above the approved critical level of {TH['vibe_crit']} m/s².")
            elif np.nanmax(p95) > TH["vibe_warn"]:
                runs = intervals(t, worst > TH["vibe_warn"], min_len=1.0)
                s, e = (runs[0][0], runs[-1][1]) if runs else fl.air_intervals()[0]
                a2 = int(np.nanargmax(p95))
                F.add("VIB_HIGH", "warning", f"Sustained vibration, {'xyz'[a2]} axis p95 {p95[a2]:.0f} m/s²", s, e, "rule:vibe_warn",
                      [ev("arjuna_vibration", axes[a2], "p95 in air", p95[a2], "m/s^2", None, inst),
                       ev("arjuna_vibration", axes[ax], "max in air", mx[ax], "m/s^2", t[peak_i], inst)],
                      basis=f"95th percentile above the approved warning level of {TH['vibe_warn']} m/s².")
        if "clip_rate" in vb and np.isfinite(vb["clip_rate"].to_numpy(float)[m]).any():
            F.ran("VIB_HIGH", "IMU clipping rate")
            c = np.where(m, vb["clip_rate"].to_numpy(float), np.nan)
            if np.nanmax(c) > TH["clip_rate_warn"]:
                i = int(np.nanargmax(c))
                runs = intervals(t, c > TH["clip_rate_warn"])
                s, e = next(((a, b) for a, b in runs if a <= t[i] <= b), (t[i], t[i]))
                s, e = (min(s, t[i] - EVENT_PAD_S / 2), max(e, t[i] + EVENT_PAD_S / 2)) if e - s < 1 else (s, e)
                F.add("VIB_HIGH", "warning", f"IMU clipping, up to {np.nanmax(c):.0f} clips/s", s, e, "rule:clip_rate",
                      [ev("arjuna_vibration", "clip_rate", "max in air", np.nanmax(c), "1/s", t[i], inst),
                       ev("arjuna_vibration", "clip_rate", "seconds above limit", float(np.sum(c > TH["clip_rate_warn"]) / max(1e-9, _rate(t))), "s", None, inst)],
                      basis=f"Above the approved limit of {TH['clip_rate_warn']} clips/s.")


def _rate(t: np.ndarray) -> float:
    d = np.diff(t)
    d = d[d > 0]
    return 1.0 / float(np.median(d)) if len(d) else 1.0


def check_attitude(fl: ArjunaFlight, F: Findings) -> None:
    at = fl.get("arjuna_attitude")
    if at is None:
        F.cannot("CRASH", "no arjuna_attitude topic")
        return
    t = at["t"].to_numpy(float)
    m = fl.in_air(t)
    F.ran("CRASH", "tilt in the air")
    if not m.any():
        return
    tilt = np.degrees(np.maximum(np.abs(at["roll"].to_numpy(float)), np.abs(at["pitch"].to_numpy(float))))
    tilt = np.where(m, tilt, np.nan)
    if np.nanmax(tilt) > TH["tilt_crit_deg"]:
        i = int(np.nanargmax(tilt))
        runs = intervals(t, tilt > TH["tilt_crit_deg"], merge_gap=1.0)
        s, e = next(((a, b) for a, b in runs if a <= t[i] <= b), (t[i], t[i]))
        F.add("CRASH", "critical", f"Tilt reached {np.nanmax(tilt):.0f}° in the air", min(s, t[i] - 1), max(e, t[i] + 1), "rule:tilt",
              [ev("arjuna_attitude", "roll" if abs(at['roll'].iloc[i]) >= abs(at['pitch'].iloc[i]) else "pitch",
                  "max tilt in air", np.nanmax(tilt), "deg", t[i])],
              basis=f"Above the approved critical tilt of {TH['tilt_crit_deg']}°.")


def _controllers(fl: ArjunaFlight, cid: int) -> List[Tuple[int, int, Any]]:
    out = []
    for inst in fl.instances("arjuna_controller"):
        df = fl.get("arjuna_controller", inst)
        if len(df) and int(df["controller_id"].iloc[0]) == cid:
            out.append((inst, int(df["axis"].iloc[0]), df))
    return out


def check_tracking(fl: ArjunaFlight, F: Findings) -> Dict[str, Any]:
    ctl = [(i, a, d) for i, a, d in _controllers(fl, 1) if a in (0, 1)]
    metrics: Dict[str, Any] = {}
    if not ctl:
        F.cannot("CTRL_OSC", "no attitude-controller records (arjuna_controller controller_id 1)")
        return metrics
    F.ran("CTRL_OSC", "attitude tracking error")
    errs = []
    for inst, axis, df in ctl:
        t = df["t"].to_numpy(float)
        m = fl.in_air(t)
        e = np.degrees(np.abs(df["error"].to_numpy(float)))
        e = np.where(m, e, np.nan)
        ok = np.isfinite(e)
        if ok.sum() < 5:
            continue
        rms = float(np.sqrt(np.nanmean(e[ok] ** 2)))
        p99 = float(np.nanpercentile(e[ok], 99))
        errs.append((axis, inst, rms, p99, t, e))
    if not errs:
        return metrics
    rms_max = max(x[2] for x in errs)
    worst = max(errs, key=lambda x: x[3])
    metrics.update(att_err_rms_deg=_r(rms_max, 2), att_err_p99_deg=_r(worst[3], 2))
    if rms_max > TH["att_err_rms_warn_deg"] or worst[3] > TH["att_err_max_warn_deg"]:
        axis, inst, rms, p99, t, e = worst
        w = max(3, int(round(RATE_WINDOW_S * _rate(t))))
        roll_rms = np.sqrt(np.convolve(np.nan_to_num(e) ** 2, np.ones(w) / w, mode="same"))
        runs = intervals(t, roll_rms > TH["att_err_rms_warn_deg"], min_len=0.5)
        i = int(np.nanargmax(e))
        s, e2 = next(((a, b) for a, b in runs if a <= t[i] <= b), (t[i] - EVENT_PAD_S, t[i] + EVENT_PAD_S))
        name = ["roll", "pitch"][axis]
        F.add("CTRL_OSC", "warning", f"Poor {name} tracking: error RMS {rms:.1f}°, p99 {p99:.0f}°", s, e2, "rule:tracking_error",
              [ev("arjuna_controller", "error", "rms in air", rms, "deg", None, inst, controller="attitude", axis=name),
               ev("arjuna_controller", "error", "p99 in air", p99, "deg", None, inst, controller="attitude", axis=name),
               ev("arjuna_controller", "error", "max in air", float(np.nanmax(e)), "deg", t[i], inst, controller="attitude", axis=name)],
              basis=f"Approved warning: RMS above {TH['att_err_rms_warn_deg']}° or p99 above {TH['att_err_max_warn_deg']}°.")
    return metrics


def check_saturation(fl: ArjunaFlight, F: Findings) -> None:
    ao = fl.get("arjuna_actuator_outputs")
    if ao is None:
        F.cannot("CTRL_SAT", "no arjuna_actuator_outputs topic")
        return
    t = ao["t"].to_numpy(float)
    m = fl.in_air(t)
    n = _motor_count(ao)
    pmax = float(fl.params.get("MOT_PWM_MAX", 2000) or 2000)
    F.ran("CTRL_SAT", "motor at maximum output")
    dt = 1.0 / _rate(t)
    for i in range(n):
        col = f"pwm_us[{i}]"
        if col not in ao:
            continue
        hi = (ao[col].to_numpy(float) >= pmax - 25) & m
        for s, e in intervals(t, hi, merge_gap=dt * 1.5):
            if e - s + dt > TH["motor_sat_s"]:
                F.add("CTRL_SAT", "warning", f"Motor {i + 1} at maximum output for {e - s + dt:.1f} s", s, e + dt, "rule:motor_sat",
                      [ev("arjuna_actuator_outputs", col, "time at max", e - s + dt, "s", s),
                       ev("arjuna_actuator_outputs", col, "max", float(np.nanmax(ao[col].to_numpy(float)[m])), "us")],
                      basis=f"PWM within 25 µs of MOT_PWM_MAX ({pmax:.0f} µs) for more than the approved {TH['motor_sat_s']} s.")
    if "mixer_saturated" in ao and np.isfinite(ao["mixer_saturated"].to_numpy(float)).any():
        sat = (np.nan_to_num(ao["mixer_saturated"].to_numpy(float)) > 0) & m
        F.ran("CTRL_SAT", "mixer saturation flags")
        for s, e in intervals(t, sat, merge_gap=dt * 1.5):
            if e - s + dt > TH["motor_sat_s"]:
                F.add("CTRL_SAT", "warning", f"Mixer saturated for {e - s + dt:.1f} s", s, e + dt, "rule:mixer_sat",
                      [ev("arjuna_actuator_outputs", "mixer_saturated", "time saturated", e - s + dt, "s", s)],
                      basis=f"Mixer reported a motor at its limit for more than {TH['motor_sat_s']} s.")


def _motor_count(ao: Any) -> int:
    v = ao["motor_count"].dropna() if "motor_count" in ao else []
    n = int(v.iloc[0]) if len(v) and 0 < v.iloc[0] < 9 else 4
    return n


def check_power(fl: ArjunaFlight, F: Findings) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    bt = fl.get("arjuna_battery")
    if bt is None or not np.isfinite(bt["voltage"].to_numpy(float)).any() or np.nanmax(bt["voltage"]) <= 5:
        F.cannot("POWER", "battery not logged on this flight")
        metrics["battery"] = "not logged"
        return metrics
    F.ran("POWER", "cell voltage and sag")
    t = bt["t"].to_numpy(float)
    v = bt["voltage"].to_numpy(float)
    m = fl.in_air(t)
    rest = float(np.nanmedian(v[~m][:50])) if (~m).sum() and np.isfinite(v[~m][:50]).any() else float(np.nanmax(v))
    cc = bt["cell_count"].dropna() if "cell_count" in bt else []
    cells = int(cc.iloc[0]) if len(cc) and 0 < cc.iloc[0] < 15 else max(1, int(round(rest / 4.0)))
    metrics["cells"] = cells
    if m.any() and np.isfinite(v[m]).any():
        va = np.where(m, v, np.nan)
        i = int(np.nanargmin(va))
        vmin = float(va[i])
        per, sag = vmin / cells, (rest - vmin) / cells
        metrics.update(min_cell_v=_r(per, 3), sag_cell_v=_r(sag, 3))
        evid = [ev("arjuna_battery", "voltage", "min in air", vmin, "V", t[i]),
                ev("arjuna_battery", "voltage", "at rest before take-off", rest, "V"),
                ev("arjuna_battery", "voltage", "min per cell", per, "V/cell", t[i], cells=cells),
                ev("arjuna_battery", "voltage", "sag per cell", sag, "V/cell", None, cells=cells)]
        if "current" in bt and np.isfinite(bt["current"].to_numpy(float)[i]):
            evid.append(ev("arjuna_battery", "current", "at minimum voltage", bt["current"].iloc[i], "A", t[i]))
        low = per < TH["cell_min_warn"]
        runs = intervals(t, np.where(m, v / cells, np.nan) < (TH["cell_min_crit"] if per < TH["cell_min_crit"] else TH["cell_min_warn"]))
        s, e = next(((a, b) for a, b in runs if a <= t[i] <= b), _pad(t[i], t[0], t[-1]))
        if per < TH["cell_min_crit"]:
            F.add("POWER", "critical", f"Battery fell to {per:.2f} V per cell", s, e, "rule:cell_min", evid,
                  basis=f"Below the approved critical {TH['cell_min_crit']} V/cell.")
        elif low or sag > TH["cell_sag_warn"]:
            F.add("POWER", "warning", f"Battery low or sagging: min {per:.2f} V/cell, sag {sag:.2f} V/cell", s, e, "rule:cell_sag", evid,
                  basis=f"Approved warning: below {TH['cell_min_warn']} V/cell or sag above {TH['cell_sag_warn']} V/cell.")
    if "rail_5v" in bt and np.isfinite(bt["rail_5v"].to_numpy(float)).any():
        F.ran("POWER", "5 V rail")
        r5 = np.where(m, bt["rail_5v"].to_numpy(float), np.nan) if m.any() else bt["rail_5v"].to_numpy(float)
        med = float(np.nanmedian(r5))
        i = int(np.nanargmin(r5))
        if med - r5[i] > TH["rail5v_dip_warn"]:
            s, e = _pad(t[i], t[0], t[-1])
            F.add("POWER", "warning", f"5 V rail dipped to {r5[i]:.2f} V (normal {med:.2f} V)", s, e, "rule:rail_5v",
                  [ev("arjuna_battery", "rail_5v", "min in air", r5[i], "V", t[i]), ev("arjuna_battery", "rail_5v", "median in air", med, "V")],
                  basis=f"More than the approved {TH['rail5v_dip_warn']} V below its own median.")
    return metrics


def check_gps(fl: ArjunaFlight, F: Findings) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    if not fl.has("arjuna_gps"):
        F.cannot("GPS_DEGRADED", "no GPS fitted or logged")
        return metrics
    for inst in fl.instances("arjuna_gps"):
        g = fl.get("arjuna_gps", inst)
        t = g["t"].to_numpy(float)
        m = fl.in_air(t)
        if not m.any():
            continue
        F.ran("GPS_DEGRADED", "fix, satellites and accuracy")
        fix = g["fix_type"].to_numpy(float)
        nofix = (fix < 2) & m
        frac = float(nofix.sum() / m.sum())
        sats = np.where(m, g["satellites"].to_numpy(float), np.nan)
        hacc = np.where(m, g["h_acc"].to_numpy(float), np.nan)
        smin = float(np.nanmin(sats)) if np.isfinite(sats).any() else None
        hmax = float(np.nanmax(hacc)) if np.isfinite(hacc).any() else None
        metrics.update(min_sats=smin, max_hacc_m=_r(hmax, 2), no_fix_frac=_r(frac, 3))
        if frac > 0.02:
            runs = intervals(t, nofix)
            F.add("GPS_DEGRADED", "warning", f"No 3D GPS fix for {frac * 100:.0f}% of air time", runs[0][0], runs[-1][1], "rule:no_fix",
                  [ev("arjuna_gps", "fix_type", "fraction of air time without 3D fix", frac, None, runs[0][0], inst)],
                  basis="More than 2% of air time without a 3D fix (approved triage).")
        elif (smin is not None and smin < TH["gps_sats_warn"]) or (hmax is not None and hmax > TH["gps_hacc_warn"]):
            bad = ((sats < TH["gps_sats_warn"]) | (hacc > TH["gps_hacc_warn"])) & m
            runs = intervals(t, bad)
            s, e = (runs[0][0], runs[-1][1]) if runs else fl.air_intervals()[0]
            evid = []
            if smin is not None:
                evid.append(ev("arjuna_gps", "satellites", "min in air", smin, "count", t[int(np.nanargmin(sats))], inst))
            if hmax is not None:
                evid.append(ev("arjuna_gps", "h_acc", "max in air", hmax, "m", t[int(np.nanargmax(hacc))], inst))
            F.add("GPS_DEGRADED", "warning", f"Weak GPS: min {smin:.0f} satellites, h_acc up to {hmax:.1f} m" if smin is not None and hmax is not None
                  else "Weak GPS", s, e, "rule:gps_quality", evid,
                  basis=f"Approved warning: fewer than {TH['gps_sats_warn']} satellites or h_acc above {TH['gps_hacc_warn']} m.")
        for fld in ("jamming_indicator", "spoofing_state"):
            if fld in g and np.isfinite(g[fld].to_numpy(float)[m]).any():
                F.ran("GPS_DEGRADED", fld)
                vals = np.where(m, g[fld].to_numpy(float), np.nan)
                if fld == "spoofing_state" and np.nanmax(vals) > 0:
                    i = int(np.nanargmax(vals))
                    F.add("GPS_DEGRADED", "critical", "GPS receiver reported spoofing", *_pad(t[i], t[0], t[-1]), "rule:spoofing",
                          [ev("arjuna_gps", fld, "max in air", vals[i], "enum", t[i], inst)], confidence="medium",
                          basis="Receiver spoofing flag raised (guide definition; no approved threshold).")
    return metrics


def check_estimator(fl: ArjunaFlight, F: Findings) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    worst = (0.0, None, None)
    for inst in fl.instances("arjuna_estimator_innov"):
        d = fl.get("arjuna_estimator_innov", inst)
        t = d["t"].to_numpy(float)
        m = fl.in_air(t)
        tr = np.where(m, d["test_ratio"].to_numpy(float), np.nan)
        if not np.isfinite(tr).any():
            continue
        F.ran("EST_DIVERGE", "innovation test ratios")
        frac = float(np.mean(tr[np.isfinite(tr)] > 1.0))
        if frac > worst[0]:
            src = d["source"].dropna()
            worst = (frac, enum_name("INNOV_SOURCE", src.iloc[0]) if len(src) else str(inst), (inst, t, tr))
    if worst[2] is not None:
        metrics["innov_over1_frac"] = _r(worst[0], 4)
        if worst[0] > TH["innov_ratio_frac_warn"]:
            inst, t, tr = worst[2]
            runs = intervals(t, tr > 1.0)
            F.add("EST_DIVERGE", "warning", f"{worst[1].replace('_', ' ')} innovations rejected {worst[0] * 100:.0f}% of air time",
                  runs[0][0], runs[-1][1], "rule:innovation_ratio",
                  [ev("arjuna_estimator_innov", "test_ratio", "fraction of air time above 1", worst[0], None, None, inst, source=worst[1]),
                   ev("arjuna_estimator_innov", "test_ratio", "max in air", float(np.nanmax(tr)), None, t[int(np.nanargmax(tr))], inst)],
                  basis=f"Test ratio above 1 for more than the approved {TH['innov_ratio_frac_warn'] * 100:.0f}% of air time.")
    ps = fl.get("arjuna_position")
    if ps is not None and "reset_counter" in ps:
        F.ran("EST_DIVERGE", "position resets")
        t = ps["t"].to_numpy(float)
        m = fl.in_air(t)
        rc = ps["reset_counter"].to_numpy(float)[m]
        rc = rc[np.isfinite(rc)]
        if rc.size and np.ptp(rc) > 2:
            tt = t[m][np.isfinite(ps["reset_counter"].to_numpy(float)[m])]
            F.add("EST_DIVERGE", "warning", f"{int(np.ptp(rc))} position estimator resets in the air", tt[0], tt[-1], "rule:position_resets",
                  [ev("arjuna_position", "reset_counter", "increase in air", np.ptp(rc), "count")],
                  basis="More than 2 resets in the air (approved triage).")
    if not F.coverage["EST_DIVERGE"]["checks"]:
        F.cannot("EST_DIVERGE", "no estimator innovation or position topic")
    return metrics


def check_sensors(fl: ArjunaFlight, F: Findings) -> None:
    sensors = [("arjuna_imu", ["gyro[0]", "gyro[1]", "gyro[2]", "accel[0]", "accel[1]", "accel[2]"]),
               ("arjuna_baro", ["pressure"]), ("arjuna_mag", ["field[0]", "field[1]", "field[2]"]),
               ("arjuna_rangefinder", ["distance"])]
    any_checked = False
    for topic, fields in sensors:
        for inst in fl.instances(topic):
            df = fl.get(topic, inst)
            t = df["t"].to_numpy(float)
            m = fl.in_air(t)
            if m.sum() < 10:
                continue
            any_checked = True
            F.ran("SENSOR_FAIL", "status, freezes and update gaps")
            if "status" in df:
                st = np.where(m, df["status"].to_numpy(float), np.nan)
                bad = st >= 2
                for s, e in intervals(t, bad):
                    F.add("SENSOR_FAIL", "critical", f"{topic.replace('arjuna_', '')} {inst} reported failed or not responding", s, e,
                          "rule:sensor_status", [ev(topic, "status", "worst in air", np.nanmax(st[(t >= s) & (t <= e)]), "enum", s, inst,
                                                    name=enum_name("SENSOR_STATUS", np.nanmax(st[(t >= s) & (t <= e)])))],
                          basis="Sensor status field: failed or not responding.")
            for f in fields:
                if f not in df:
                    continue
                y = df[f].to_numpy(float)
                same = np.concatenate([[False], (np.diff(y) == 0) & np.isfinite(y[1:])]) & m
                for s, e in intervals(t, same, merge_gap=1.5 / _rate(t)):
                    n = int(np.sum(same & (t >= s) & (t <= e)))
                    if e - s >= FREEZE_MIN_S and n >= 10:
                        F.add("SENSOR_FAIL", "warning", f"{topic.replace('arjuna_', '')} {f} frozen for {e - s:.1f} s", s, e,
                              "rule:sensor_freeze", [ev(topic, f, "identical consecutive samples", n + 1, "count", s, inst),
                                                     ev(topic, f, "frozen value", y[np.argmax((t >= s))], None, s, inst)],
                              confidence="medium", basis=f"Guide definition (sensor freezes); constant for at least {FREEZE_MIN_S} s in the air. "
                              "Threshold not yet approved.")
                        break
            dt = np.diff(t)
            med = float(np.median(dt[dt > 0])) if (dt > 0).any() else 0
            if med > 0:
                gap = np.where((dt > UPDATE_GAP_FACTOR * med) & m[1:])[0]
                if len(gap):
                    k = int(gap[np.argmax(dt[gap])])
                    F.add("SENSOR_FAIL", "warning", f"{topic.replace('arjuna_', '')} {inst} stopped updating for {dt[k]:.2f} s", t[k], t[k + 1],
                          "rule:update_gap", [ev(topic, "t", "longest gap in air", dt[k], "s", t[k], inst),
                                              ev(topic, "t", "median period", med, "s", None, inst),
                                              ev(topic, "t", "gaps in air", len(gap), "count", None, inst)],
                          confidence="medium", basis=f"Guide definition (sensor stops); a gap over {UPDATE_GAP_FACTOR:.0f}x the median period. "
                          "Threshold not yet approved.")
            if "error_count" in df and np.isfinite(df["error_count"].to_numpy(float)).any():
                ec = df["error_count"].to_numpy(float)
                inc = float(np.nanmax(ec[m]) - np.nanmin(ec[m])) if np.isfinite(ec[m]).any() else 0
                if inc > 0:
                    i = int(np.argmax(np.nan_to_num(np.diff(ec)) > 0)) + 1
                    F.add("SENSOR_FAIL", "warning", f"{topic.replace('arjuna_', '')} {inst} bus errors in the air (+{inc:.0f})", t[i], t[-1],
                          "rule:bus_errors", [ev(topic, "error_count", "increase in air", inc, "count", t[i], inst)], confidence="medium",
                          basis="Bus read errors increased during flight (guide definition; no approved threshold).")
    if not any_checked:
        F.cannot("SENSOR_FAIL", "no sensor topics with air-time samples")


def check_timing(fl: ArjunaFlight, F: Findings) -> None:
    if not fl.has("arjuna_fpga_timing"):
        F.cannot("TIMING", "arjuna_fpga_timing not in this log (converted logs cannot produce it)")
    for inst in fl.instances("arjuna_fpga_timing"):
        d = fl.get("arjuna_fpga_timing", inst)
        F.ran("TIMING", "deadline misses, overruns, FIFO errors, data age")
        t = d["t"].to_numpy(float)
        for fld, label in (("deadline_misses", "deadline misses"), ("overruns", "overruns"),
                           ("fifo_overflows", "FIFO overflows"), ("fifo_underflows", "FIFO underflows")):
            if fld in d:
                y = np.nan_to_num(d[fld].to_numpy(float))
                if y.sum() > 0:
                    runs = intervals(t, y > 0)
                    F.add("TIMING", "warning", f"Task {inst}: {int(y.sum())} {label}", runs[0][0], runs[-1][1], f"rule:{fld}",
                          [ev("arjuna_fpga_timing", fld, "total", y.sum(), "count", runs[0][0], inst)], confidence="medium",
                          basis="Any late or overlapping run is a timing fault (guide definition; no approved threshold).")
        if "data_age_max" in d and "period_us" in d:
            age, per = d["data_age_max"].to_numpy(float), d["period_us"].to_numpy(float)
            stale = age > 2 * per
            if stale.any():
                i = int(np.nanargmax(age))
                F.add("TIMING", "warning", f"Task {inst}: stale input data up to {age[i]:.0f} µs", *_pad(t[i], t[0], t[-1]),
                      "rule:data_age", [ev("arjuna_fpga_timing", "data_age_max", "max", age[i], "us", t[i], inst),
                                        ev("arjuna_fpga_timing", "period_us", "configured period", per[i], "us", None, inst)],
                      confidence="medium", basis="Input older than two task periods (guide definition; no approved threshold).")
    for inst, axis, df in _controllers(fl, 0) + _controllers(fl, 1):
        if "deadline_missed" in df and np.nansum(df["deadline_missed"].to_numpy(float)) > 0:
            F.ran("TIMING", "controller deadlines")
            t = df["t"].to_numpy(float)
            miss = df["deadline_missed"].to_numpy(float) > 0
            runs = intervals(t, miss)
            F.add("TIMING", "warning", f"Controller steps finished late ({int(miss.sum())} times)", runs[0][0], runs[-1][1],
                  "rule:controller_deadline", [ev("arjuna_controller", "deadline_missed", "count", miss.sum(), "count", runs[0][0], inst)],
                  confidence="medium", basis="Controller step finished after its deadline.")


def check_comms(fl: ArjunaFlight, F: Findings) -> None:
    rc = fl.get("arjuna_rc_input")
    if rc is not None and "failsafe" in rc and np.isfinite(rc["failsafe"].to_numpy(float)).any():
        F.ran("COMMS", "RC receiver failsafe")
        t = rc["t"].to_numpy(float)
        fs = (rc["failsafe"].to_numpy(float) > 0) & fl.in_air(t)
        for s, e in intervals(t, fs, merge_gap=1.0):
            F.add("COMMS", "critical", f"RC signal lost for {e - s:.1f} s in the air", s, e, "rule:rc_failsafe",
                  [ev("arjuna_rc_input", "failsafe", "duration in air", e - s, "s", s)], basis="Receiver reported signal loss.")
    for inst in fl.instances("arjuna_communication"):
        d = fl.get("arjuna_communication", inst)
        t = d["t"].to_numpy(float)
        m = fl.in_air(t)
        if "lost_packets" in d and np.isfinite(d["lost_packets"].to_numpy(float)[m]).any():
            F.ran("COMMS", "packet loss")
            lp = d["lost_packets"].to_numpy(float)
            rx = d["rx_packets"].to_numpy(float) if "rx_packets" in d else np.full_like(lp, np.nan)
            dl = np.nanmax(lp[m]) - np.nanmin(lp[m])
            dr = (np.nanmax(rx[m]) - np.nanmin(rx[m])) if np.isfinite(rx[m]).any() else np.nan
            if dl > 0 and np.isfinite(dr) and dl / max(1.0, dl + dr) > 0.05:
                F.add("COMMS", "warning", f"Link {inst}: {dl / (dl + dr) * 100:.0f}% packet loss in the air", *fl.air_intervals()[0],
                      "rule:packet_loss", [ev("arjuna_communication", "lost_packets", "lost in air", dl, "count", None, inst),
                                           ev("arjuna_communication", "rx_packets", "received in air", dr, "count", None, inst)],
                      confidence="medium", basis="More than 5% packet loss (guide definition; no approved threshold).", window="whole_flight")
    if not F.coverage["COMMS"]["checks"]:
        F.cannot("COMMS", "no link telemetry with usable values")


def check_security(fl: ArjunaFlight, F: Findings) -> None:
    fw = fl.get("arjuna_firewall")
    if fw is None:
        F.cannot("SECURITY", "arjuna_firewall not in this log (converted logs cannot produce it)")
        return
    F.ran("SECURITY", "rejected commands and signing")
    t = fw["t"].to_numpy(float)
    if "cmd_rejected" in fw:
        rj = fw["cmd_rejected"].to_numpy(float)
        if np.isfinite(rj).any() and np.nanmax(rj) - np.nanmin(rj) > 0:
            i = int(np.argmax(np.nan_to_num(np.diff(rj)) > 0)) + 1
            reason = enum_name("REJECT_REASON", fw["last_reject_reason"].iloc[i]) if "last_reject_reason" in fw else "unknown"
            F.add("SECURITY", "warning", f"{np.nanmax(rj) - np.nanmin(rj):.0f} commands rejected by the firewall", t[i], t[-1],
                  "rule:cmd_rejected", [ev("arjuna_firewall", "cmd_rejected", "increase", np.nanmax(rj) - np.nanmin(rj), "count", t[i], reason=reason)],
                  confidence="medium", basis="Any rejected command is reported (guide definition).")
    if "signing_active" in fw and np.isfinite(fw["signing_active"].to_numpy(float)).any() and np.nanmin(fw["signing_active"]) < 1:
        F.add("SECURITY", "warning", "Message signing was not enforced for part of the flight", t[0], t[-1], "rule:signing_off",
              [ev("arjuna_firewall", "signing_active", "min", np.nanmin(fw["signing_active"]), None)], confidence="medium",
              basis="signing_active false.", window="whole_flight")


def check_logging(fl: ArjunaFlight, F: Findings) -> None:
    F.ran("LOGGING", "dropout records")
    if fl.dropouts:
        tot = sum(d["duration_ms"] for d in fl.dropouts)
        F.add("LOGGING", "warning", f"{len(fl.dropouts)} log dropouts, {tot} ms of data lost", fl.dropouts[0]["t"], fl.dropouts[-1]["t"],
              "rule:dropouts", [ev("log", "dropout", "count", len(fl.dropouts), "count", fl.dropouts[0]["t"]),
                                ev("log", "dropout", "total duration", tot, "ms")], basis="ULog dropout records (approved triage).")
    sh = fl.get("arjuna_system_health")
    if sh is not None and "log_buffer_fill" in sh and np.isfinite(sh["log_buffer_fill"].to_numpy(float)).any():
        F.ran("LOGGING", "logger buffer fill")
        y = sh["log_buffer_fill"].to_numpy(float)
        if np.nanmax(y) >= 90:
            i = int(np.nanargmax(y))
            F.add("LOGGING", "warning", f"Logger buffer reached {y[i]:.0f}%", *_pad(sh["t"].iloc[i], 0, fl.duration_s), "rule:buffer_fill",
                  [ev("arjuna_system_health", "log_buffer_fill", "max", y[i], "%", sh["t"].iloc[i])], confidence="medium",
                  basis="Near 100% means dropouts are coming (guide).")


def check_esc(fl: ArjunaFlight, F: Findings) -> None:
    es = fl.get("arjuna_esc_status")
    if es is None:
        return
    F.ran("MOTOR_ESC", "ESC telemetry")
    t = es["t"].to_numpy(float)
    if "fault" in es:
        fb = np.nan_to_num(es["fault"].to_numpy(float)).astype(int)
        if (fb > 0).any():
            i = int(np.argmax(fb > 0))
            motors = [k + 1 for k in range(8) if fb[i] & (1 << k)]
            F.add("MOTOR_ESC", "critical", f"ESC fault reported on motor {', '.join(map(str, motors))}", t[i], t[-1], "rule:esc_fault",
                  [ev("arjuna_esc_status", "fault", "first fault bits", fb[i], "bitflags", t[i])], basis="ESC fault flags.")


# ---- driver ----------------------------------------------------------------------------------------------------------

def _source_limit(fl: ArjunaFlight, key: str, default: float) -> float:
    """Per-source limits a CSV conversion may set (e.g. short FLT test hops); ARJUNA logs use the approved values."""
    try:
        return float(fl.info.get(key, default))
    except (TypeError, ValueError):
        return default


def _height(fl: ArjunaFlight):
    for topic, fld in (("arjuna_baro", "altitude"), ("arjuna_rangefinder", "distance"), ("arjuna_position", "pos_u")):
        df = fl.get(topic)
        if df is not None and fld in df and np.isfinite(df[fld].to_numpy(float)).any():
            return f"{topic}.{fld}", df["t"].to_numpy(float), df[fld].to_numpy(float)
    return None


def classify(fl: ArjunaFlight) -> Dict[str, Any]:
    air = round(fl.airborne_s(), 1)  # rounded like the approved triage, so boundary flights classify the same
    min_air = _source_limit(fl, "min_airborne_s", TH["min_airborne_s"])
    min_alt = _source_limit(fl, "min_alt_range_m", 2.0)
    kind, why = "flight", ""
    if air < min_air:
        kind, why = "ground_only", f"in the air for {air:.1f} s (under {min_air:g} s)"
    else:
        at, h = fl.get("arjuna_attitude"), _height(fl)
        if at is not None:
            m = fl.in_air(at["t"].to_numpy(float))
            tilt = float(np.degrees(np.nanmax(np.abs(np.concatenate([at["roll"].to_numpy(float)[m], at["pitch"].to_numpy(float)[m]]))))) if m.any() else 0.0
            altr = 0.0
            if h is not None:
                al = h[2][fl.in_air(h[1])]
                altr = float(np.nanpercentile(al, 98) - np.nanpercentile(al, 2)) if np.isfinite(al).any() else 0.0
            if tilt < 5 and altr < min_alt:
                kind, why = "bench_test", f"motors ran but the vehicle did not move (tilt {tilt:.1f}°, altitude range {altr:.1f} m)"
    return {"kind": kind, "reason": why, "airborne_s": round(air, 1), "min_airborne_s": min_air,
            "limits_from": "source conversion" if "min_airborne_s" in fl.info else "approved thresholds"}


SEV_RANK = {"critical": 0, "warning": 1, "info": 2}


def run_checks(fl: ArjunaFlight) -> Dict[str, Any]:
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices are expected for unlogged sensors
        return _run_checks(fl)


def _run_checks(fl: ArjunaFlight) -> Dict[str, Any]:
    F = Findings(fl)
    cls = classify(fl)
    airs = fl.air_intervals()
    a0, a1 = (airs[0][0], airs[-1][1]) if airs else (0.0, fl.duration_s)
    t_takeoff = airs[0][0] if airs else None
    metrics: Dict[str, Any] = {}
    check_events(fl, F, a0, a1, t_takeoff)
    check_logging(fl, F)
    if cls["kind"] == "flight":
        check_failsafe(fl, F)
        check_vibration(fl, F)
        check_attitude(fl, F)
        metrics.update(check_tracking(fl, F))
        check_saturation(fl, F)
        metrics["power"] = check_power(fl, F)
        metrics["gps"] = check_gps(fl, F)
        metrics.update(check_estimator(fl, F))
        check_sensors(fl, F)
        check_timing(fl, F)
        check_comms(fl, F)
        check_security(fl, F)
        check_esc(fl, F)
    else:
        for c, cov in F.coverage.items():
            if cov["status"] != "evaluated":
                F.cannot(c, f"not a flight: {cls['reason']}")
    items = sorted(F.items, key=lambda f: (SEV_RANK.get(f["severity"], 3), f["t_start"] if f["t_start"] is not None else 1e12))
    # de-duplicate identical (code, title) pairs, keeping the most severe
    seen, out = set(), []
    for f in items:
        k = (f["code"], f["title"])
        if k not in seen:
            seen.add(k)
            out.append(f)
    for i, f in enumerate(out, 1):
        f["id"] = f"F{i}"
    for c, cov in F.coverage.items():
        if c in F.events_only:
            cov["checks"].insert(0, "flight events")
        if cov["status"] == "evaluated":
            cov["reasons"] = []
        elif c in F.events_only:
            cov["status"] = "partial"  # only autopilot events could be read; the signal checks could not run
    return {"classification": cls, "findings": out, "coverage": F.coverage, "metrics": metrics,
            "thresholds": TH, "fault_codes": FAULT_CODES}


def worst_severity(findings: List[Dict[str, Any]]) -> Optional[str]:
    sev = {f["severity"] for f in findings}
    return "critical" if "critical" in sev else "warning" if "warning" in sev else None
