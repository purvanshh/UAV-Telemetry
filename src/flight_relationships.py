"""Context-aware telemetry analysis for the AI Analysis pipeline.

This module answers the question an engineer actually asks: *is the recorded behaviour
consistent with what the aircraft was commanded to do, with its own sensors, and with
gravity?* It never scores or thresholds events; it measures relationships and states how
well the evidence supports each one.

Everything is computed in physical units **only where the central schema marks the unit as
verified**. Otherwise the raw value is used and the metric is labelled raw. Axis/sign
conventions are not established for this dataset, so a command-response relationship learns
its direction from the recording itself (sign of the correlation) instead of assuming that
"positive stick" means "positive roll".

Output shape (all JSON-serialisable):
``analyse_flight`` returns timing, data quality, per-category signal inventory, relationships
and a plain-language list of what the data can and cannot say. ``annotate_events`` adds the
command/consistency interpretation to each detected event.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.telemetry_schema import ColumnSpec, DatasetSchema

MIN_CORR_SAMPLES = 30
CONSISTENT_R = 0.6
PARTIAL_R = 0.35
BLOCK_S = 0.2
MIN_BALANCE_FRACTION = 0.05
GRAVITY = 9.80665
GRAVITY_TOLERANCE = 1.0

CATEGORY_TITLES = {
    "attitude": "Attitude",
    "acceleration": "Acceleration",
    "angular_rate": "Angular rate",
    "altitude": "Altitude",
    "distance": "Distance",
    "motor": "Motor output",
    "command": "Pilot input",
    "control": "Controller terms",
    "status": "Status flags",
    "aux": "Auxiliary channels",
}

INTERPRETATION_LABEL = {
    "expected_response": "Consistent with pilot input",
    "unexplained_deviation": "Not explained by the recorded input",
    "telemetry_inconsistency": "Sensors disagree",
    "control_input_change": "Pilot or command input change",
    "status_change": "Status flag change",
    "data_quality": "Recording problem",
    "insufficient_evidence": "Not enough evidence",
}


def _r(x: Any, nd: int = 4) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if math.isfinite(v) else None


def _mad(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else 0.0


def _corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < MIN_CORR_SAMPLES or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return None
    v = float(np.corrcoef(a[ok], b[ok])[0, 1])
    return v if math.isfinite(v) else None


def _blocks(x: np.ndarray, size: int) -> np.ndarray:
    n = len(x) // size
    if n == 0:
        return np.array([])
    return np.nanmedian(x[: n * size].reshape(n, size), axis=1)


def _best_lag(x: np.ndarray, y: np.ndarray, max_lag: int) -> Tuple[Optional[float], int, int]:
    """Best |correlation| between block series at lags -max_lag..max_lag; returns (r, lag, n)."""
    best: Tuple[Optional[float], int, int] = (None, 0, 0)
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a, b = x[-lag:], y[: len(y) + lag]
        elif lag > 0:
            a, b = x[: len(x) - lag], y[lag:]
        else:
            a, b = x, y
        r = _corr(a, b)
        if r is not None and (best[0] is None or abs(r) > abs(best[0])):
            best = (r, lag, int((np.isfinite(a) & np.isfinite(b)).sum()))
    return best


def _status_from_corr(r: Optional[float], enough: bool) -> str:
    if r is None:
        return "insufficient_data" if not enough else "no_relationship"
    if abs(r) >= CONSISTENT_R:
        return "consistent"
    if abs(r) >= PARTIAL_R:
        return "partially_consistent"
    return "not_consistent"


class _Access:
    """Column access by canonical name for one table."""

    def __init__(self, table: pd.DataFrame, canonical_by_header: Dict[str, str]):
        self.table = table
        self.by_canonical = {c: h for h, c in canonical_by_header.items() if h in table}

    def has(self, canonical: str) -> bool:
        return canonical in self.by_canonical

    def get(self, canonical: str) -> Optional[np.ndarray]:
        h = self.by_canonical.get(canonical)
        if h is None:
            return None
        return self.table[h].to_numpy(float)

    def header(self, canonical: str) -> Optional[str]:
        return self.by_canonical.get(canonical)


def _unit_of(specs: Dict[str, ColumnSpec], canonical: str) -> Optional[str]:
    spec = specs.get(canonical)
    return spec.display_unit if spec else None


def _fmt_value(value: Any, unit: Optional[str], digits: int = 2) -> Optional[str]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    text = f"{v:,.{digits}f}".rstrip("0").rstrip(".")
    return f"{text} {unit}" if unit else text


def analyse_flight(
    table: pd.DataFrame,
    *,
    specs: Dict[str, ColumnSpec],
    canonical_by_header: Dict[str, str],
    time_s: Optional[np.ndarray],
    dt_s: Optional[float],
    time_verified: bool,
    gaps: Sequence[Dict[str, Any]] = (),
    headers: Sequence[str] = (),
) -> Dict[str, Any]:
    """Measure timing, data quality, signal inventory and inter-signal relationships."""
    a = _Access(table, canonical_by_header)
    n = len(table)
    out: Dict[str, Any] = {}

    # ---- timing -------------------------------------------------------------
    duration = float(time_s[-1] - time_s[0]) if time_s is not None and len(time_s) else None
    dt_vals = np.diff(time_s) if time_s is not None else np.array([])
    dt_ok = dt_vals[np.isfinite(dt_vals) & (dt_vals > 0)]
    timing = {
        "duration_s": _r(duration),
        "time_verified": bool(time_verified),
        "median_step_s": _r(np.median(dt_ok)) if len(dt_ok) else _r(dt_s),
        "step_p05_s": _r(np.percentile(dt_ok, 5)) if len(dt_ok) else None,
        "step_p95_s": _r(np.percentile(dt_ok, 95)) if len(dt_ok) else None,
        "sample_rate_hz": _r(1.0 / float(np.median(dt_ok)), 2) if len(dt_ok) and np.median(dt_ok) > 0 else None,
        "n_samples": int(n),
        "n_gaps": len(gaps),
        "irregular_sampling": bool(len(dt_ok) and np.percentile(dt_ok, 95) > 1.5 * np.percentile(dt_ok, 5)),
        "short_recording": bool(duration is not None and duration < 3.0),
    }
    out["timing"] = timing

    # ---- data quality -------------------------------------------------------
    missing = {}
    out_of_range = []
    for h, c in canonical_by_header.items():
        if h not in table:
            continue
        y = table[h].to_numpy(float)
        n_missing = int(np.isnan(y).sum())
        if n_missing:
            missing[h] = n_missing
        spec = specs.get(c)
        if spec and spec.physical_range and np.isfinite(y).any():
            lo, hi = spec.physical_range
            bad = int(((y < lo) | (y > hi)).sum())
            if bad:
                out_of_range.append({"channel": h, "canonical": c, "count": bad,
                                     "expected_range": [lo, hi]})
    out["data_quality"] = {"missing_values": missing, "out_of_range": out_of_range,
                           "constant_channels": [h for h, c in canonical_by_header.items()
                                                 if h in table and table[h].nunique(dropna=True) <= 1]}

    # ---- signal inventory ---------------------------------------------------
    present = {c for c in canonical_by_header.values()}
    inventory: Dict[str, List[Dict[str, Any]]] = {}
    for canonical, spec in specs.items():
        if canonical not in present:
            continue
        inventory.setdefault(spec.category, []).append({
            "canonical": canonical, "column": a.header(canonical), "role": spec.role,
            "unit": spec.display_unit, "unit_status": spec.unit_status,
        })
    out["signals"] = {
        "by_category": inventory,
        "available": sorted(present),
        "missing_declared": sorted(c for c in specs if c not in present),
    }

    # ---- smoothing block series (manoeuvre scale) ---------------------------
    block = max(1, int(round(BLOCK_S / dt_s))) if dt_s else 5
    rels: Dict[str, Any] = {}

    def block_pair(x: Optional[np.ndarray], y: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
        if x is None or y is None or len(x) != len(y):
            return None, None, 0
        return _blocks(x, block), _blocks(y, block), block

    def command_relationship(rid: str, label: str, command: Optional[str],
                             response: Optional[str], *, threshold: float = CONSISTENT_R) -> Optional[Dict[str, Any]]:
        if not command or not response or not a.has(command) or not a.has(response):
            return None
        cb, rb, b = block_pair(a.get(command), a.get(response))
        if cb is None or len(cb) < 8:
            return None
        raw_cmd = a.get(command)
        cmd_range = float(np.nanpercentile(raw_cmd, 95) - np.nanpercentile(raw_cmd, 5)) if raw_cmd is not None else 0.0
        need = (specs.get(command).min_excitation if specs.get(command) else None)
        excited = cmd_range >= (need if need is not None else 0.0)
        max_lag = max(1, min(5, len(cb) // 8))
        r, lag, npts = _best_lag(cb, rb, max_lag)
        enough = excited and npts >= 8 and np.nanstd(cb) > 0 and np.nanstd(rb) > 0
        status = _status_from_corr(r, enough)
        if not excited:
            status = "insufficient_data"
        return {
            "label": label, "kind": "command_response", "command": command, "response": response,
            "status": status, "r": _r(r, 3), "lag_blocks": int(lag),
            "command_range_raw": _r(cmd_range), "min_excitation": need, "excited": bool(excited),
            "lag_s": _r(lag * BLOCK_S, 2), "n_blocks": int(npts),
            "direction_sign": (1 if (r or 0) >= 0 else -1) if r is not None else None,
            "block_s": BLOCK_S,
            "detail": ("Relationship between the recorded pilot input and the aircraft response over the whole "
                       "recording. Direction is learned from the data; it is not an assumed sign convention."),
        }

    # A. roll / pitch command vs attitude
    for rid, label, cmd, resp in (
        ("roll_tracking", "Roll command vs roll angle", "roll_cmd", "roll"),
        ("pitch_tracking", "Pitch command vs pitch angle", "pitch_cmd", "pitch"),
    ):
        rel = command_relationship(rid, label, cmd, resp)
        if rel:
            rel["unit"] = _unit_of(specs, resp)
            rels[rid] = rel

    # B. throttle vs motors
    mot = [c for c in ("motor_1", "motor_2", "motor_3", "motor_4") if a.has(c)]
    if a.has("throttle_cmd") and mot:
        mean_motor = np.nanmean(np.vstack([a.get(c) for c in mot]), axis=0)
        cb, rb, _ = block_pair(a.get("throttle_cmd"), mean_motor)
        r, lag, npts = _best_lag(cb, rb, max(1, min(5, len(cb) // 8))) if cb is not None and len(cb) >= 8 else (None, 0, 0)
        raw_thr = a.get("throttle_cmd")
        thr_range = float(np.nanpercentile(raw_thr, 95) - np.nanpercentile(raw_thr, 5)) if raw_thr is not None else 0.0
        need = specs.get("throttle_cmd").min_excitation if specs.get("throttle_cmd") else None
        excited = thr_range >= (need if need is not None else 0.0)
        enough = excited and cb is not None and npts >= 8 and np.nanstd(cb) > 0 and np.nanstd(rb) > 0
        rels["throttle_motors"] = {
            "label": "Throttle command vs motor outputs", "kind": "command_response",
            "command": "throttle_cmd", "response": "motor_mean", "channels": mot,
            "status": "insufficient_data" if not excited else _status_from_corr(r, enough),
            "r": _r(r, 3), "lag_blocks": int(lag), "lag_s": _r(lag * BLOCK_S, 2),
            "command_range_raw": _r(thr_range), "min_excitation": need, "excited": bool(excited),
            "direction_sign": (1 if (r or 0) >= 0 else -1) if r is not None else None,
            "detail": "The four motor outputs are averaged and compared with the throttle input at manoeuvre scale.",
        }

    # C. altitude: estimate vs setpoint vs independent height
    if a.has("alt_est"):
        for other, kind in (("alt_set", "setpoint"), ("tof_height", "independent_measurement")):
            if not a.has(other):
                continue
            eb, ob, _ = block_pair(a.get("alt_est"), a.get(other))
            r = _corr(eb, ob) if eb is not None and len(eb) >= 8 else None
            both = np.isfinite(a.get("alt_est")) & np.isfinite(a.get(other))
            if both.sum() >= 10:
                evals, ovals = a.get("alt_est")[both], a.get(other)[both]
                scale = float(np.percentile(np.abs(np.r_[evals, ovals]), 95) or 1.0)
                level_gap = float(np.median(np.abs(evals - ovals))) / max(scale, 1e-9)
            else:
                level_gap = None
            rels[f"altitude_{kind}"] = {
                "label": f"Altitude estimate vs {'setpoint' if kind == 'setpoint' else 'time-of-flight height'}",
                "kind": kind, "response": "alt_est", "other": other, "status": _status_from_corr(r, bool(both.sum() >= 10)),
                "r": _r(r, 3), "relative_level_gap": _r(level_gap, 3),
                "agreement": ("close" if level_gap is not None and level_gap <= 0.15 else
                              "different levels" if level_gap is not None else "not assessed"),
                "detail": ("Correlation of the two altitude series at manoeuvre scale, plus the typical absolute "
                           "difference relative to their own 95th-percentile magnitude. Scale/unit are not verified, "
                           "so values stay raw."),
            }

    # D. attitude change vs angular rate
    if time_s is not None and a.has("roll") and a.has("gyro_x"):
        rels["attitude_rate_x"] = _rate_consistency(a, "roll", "gyro_x", time_s, "deg")
    if time_s is not None and a.has("pitch") and a.has("gyro_y"):
        rels["attitude_rate_y"] = _rate_consistency(a, "pitch", "gyro_y", time_s, "deg")

    # E. acceleration vs gravity / attitude
    if all(a.has(c) for c in ("acc_x", "acc_y", "acc_z")):
        mag = np.sqrt(sum(np.square(a.get(c)) for c in ("acc_x", "acc_y", "acc_z")))
        unit = _unit_of(specs, "acc_x")
        rest = _rest_mask(a, canonical_by_header, table)
        rest_mag = mag[rest & np.isfinite(mag)] if rest is not None else np.array([])
        rels["gravity"] = {
            "label": "Acceleration magnitude vs gravity",
            "kind": "physical_consistency",
            "unit": unit,
            "status": ("consistent" if len(rest_mag) >= 10 and abs(float(np.median(rest_mag)) - GRAVITY) <= GRAVITY_TOLERANCE
                       else "not_consistent" if len(rest_mag) >= 10 else "insufficient_data"),
            "rest_median": _r(np.median(rest_mag)) if len(rest_mag) else None,
            "rest_p05": _r(np.percentile(rest_mag, 5)) if len(rest_mag) else None,
            "rest_p95": _r(np.percentile(rest_mag, 95)) if len(rest_mag) else None,
            "rest_samples": int(len(rest_mag)),
            "dynamic_max": _r(np.nanmax(mag)) if np.isfinite(mag).any() else None,
            "detail": ("Norm of the three acceleration channels. At rest and near-level the norm should be about "
                       "9.8 m/s^2 including gravity; ordinary gravity is therefore never treated as an anomaly."),
        }

    # F. motor balance
    if len(mot) >= 2:
        stack = np.vstack([a.get(c) for c in mot])
        mean = np.nanmean(stack, axis=0)
        spread = np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(np.abs(mean) > 1e-9, spread / np.abs(mean), np.nan)
        f = frac[np.isfinite(frac)]
        base = float(np.median(f)) if len(f) else 0.0
        hard = np.isfinite(frac) & (frac > max(MIN_BALANCE_FRACTION, 3 * base))
        rels["motor_balance"] = {
            "label": "Balance between motor outputs", "kind": "cross_channel",
            "channels": mot, "status": "consistent" if not hard.any() else "partially_consistent",
            "median_spread_fraction": _r(base, 3), "p95_spread_fraction": _r(np.percentile(f, 95), 3) if len(f) else None,
            "max_spread_fraction": _r(np.nanmax(f), 3) if len(f) else None,
            "samples_above_threshold": int(hard.sum()),
            "detail": ("Spread between the highest and lowest motor output, as a fraction of their mean. An "
                       "increase of all motors together is not treated as imbalance."),
        }

    out["relationships"] = rels

    # ---- status transitions (reported separately; meanings not assumed) -----
    transitions = []
    for canonical in ("armed", "kill", "rc_failsafe", "tof_ok", "tof_status", "block_flag", "aux_1"):
        y = a.get(canonical)
        if y is None:
            continue
        ok = np.isfinite(y)
        idx = np.where(ok[1:] & ok[:-1] & (np.diff(y) != 0))[0]
        for i in idx[:20]:
            transitions.append({
                "channel": a.header(canonical), "canonical": canonical,
                "from": _r(y[i]), "to": _r(y[i + 1]),
                "time_s": _r(time_s[i + 1]) if time_s is not None else None,
            })
    out["status_transitions"] = transitions

    # ---- per-category summary ----------------------------------------------
    cats = {}
    for cat, items in inventory.items():
        rel_ids = [rid for rid, r in rels.items() if _relationship_categories(r, specs) == cat]
        cats[cat] = {"title": CATEGORY_TITLES.get(cat, cat.title()), "signals": [i["canonical"] for i in items],
                     "relationships": rel_ids, "events": 0}
    out["categories"] = cats

    out["sequence"] = _sequence(a, time_s, timing)
    out["limits"] = _limits(timing, rels, canonical_by_header, specs)
    return out


def _sequence(a: _Access, time_s: Optional[np.ndarray], timing: Dict[str, Any]) -> List[str]:
    """Plain, grounded sequence of the recording: arming, throttle input, motor level, altitude setpoint."""
    if time_s is None:
        return []
    out: List[str] = []
    t = time_s

    def at(idx: int) -> str:
        return f"{int(t[idx] // 60)}:{int(round(t[idx] % 60)):02d}"

    armed = a.get("armed")
    if armed is not None:
        vals = set(np.unique(armed[np.isfinite(armed)]).tolist())
        if vals and vals <= {0.0, 1.0}:
            first = float(armed[0])
            change = np.flatnonzero(np.diff(armed) != 0)
            parts = [f"Arming status starts at {int(first)}"]
            for i in change[:4]:
                parts.append(f"changes to {int(armed[i + 1])} at {at(i + 1)}")
            out.append("; ".join(parts) + ".")

    thr = a.get("throttle_cmd")
    if thr is not None:
        ok = np.isfinite(thr)
        idx = [i for i in np.flatnonzero(ok[1:] & ok[:-1] & (np.abs(np.diff(thr)) >= 50))][:3]
        for i in idx:
            out.append(f"Throttle input changes from {thr[i]:g} to {thr[i + 1]:g} at {at(i + 1)} (raw command units).")

    motors = [a.get(c) for c in ("motor_1", "motor_2", "motor_3", "motor_4") if a.has(c)]
    if motors:
        mean = np.nanmean(np.vstack(motors), axis=0)
        base = float(np.nanmin(mean))
        up = np.flatnonzero(np.isfinite(mean) & (mean > base + 25))
        if len(up):
            out.append(f"Motor output leaves its idle level ({base:g}) at {at(int(up[0]))} and reaches "
                       f"{np.nanmax(mean):.0f} by the end of the recording (raw output counts).")

    alt_set = a.get("alt_set")
    if alt_set is not None:
        ok = np.isfinite(alt_set)
        idx = np.flatnonzero(ok[1:] & ok[:-1] & (np.abs(np.diff(alt_set)) >= 0.05))[:3]
        for i in idx:
            out.append(f"Altitude setpoint changes from {alt_set[i]:g} to {alt_set[i + 1]:g} at {at(i + 1)} "
                       "(raw units; scale not verified).")
    return out


def _relationship_categories(rel: Dict[str, Any], specs: Dict[str, ColumnSpec]) -> str:
    for key in ("response", "other"):
        v = rel.get(key)
        if isinstance(v, str) and v in specs:
            return specs[v].category
    return "other"


def _rate_consistency(a: _Access, attitude: str, rate: str, time_s: np.ndarray, unit: str) -> Dict[str, Any]:
    y = a.get(attitude)
    g = a.get(rate)
    der = np.full(len(y), np.nan)
    ok = np.isfinite(y) & np.isfinite(time_s)
    idx = np.flatnonzero(ok)
    if len(idx) >= 3:
        # central difference only where the time base is contiguous
        d = np.diff(time_s)
        med = np.median(d[np.isfinite(d) & (d > 0)])
        good = np.isfinite(d) & (d > 0) & (d < 3 * med)
        der[1:] = np.where(good, np.diff(y) / np.where(d > 0, d, np.nan), np.nan)
    r = _corr(der, g)
    pair = np.isfinite(der) & np.isfinite(g)
    slope = None
    if pair.sum() >= MIN_CORR_SAMPLES and np.std(g[pair]) > 0:
        slope = float(np.polyfit(g[pair], der[pair], 1)[0])
    status = _status_from_corr(r, pair.sum() >= MIN_CORR_SAMPLES)
    if r is None and pair.sum() >= MIN_CORR_SAMPLES:
        d_std = float(np.nanstd(der[pair]))
        g_std = float(np.nanstd(g[pair]))
        # One series moves and the other does not: the two records disagree rather than being uninformative.
        if (d_std == 0.0) != (g_std == 0.0):
            status = "not_consistent"
    if status == "consistent" and slope is not None and not (0.5 <= slope <= 2.0):
        status = "partially_consistent"
    axis = str(rate).replace("gyro_", "").upper()
    return {
        "label": f"Change in {attitude} vs rotation rate about {axis}", "kind": "physical_consistency",
        "attitude": attitude, "rate": rate, "unit": unit,
        "status": status, "r": _r(r, 3), "slope": _r(slope, 3), "n": int(pair.sum()),
        "detail": ("The per-sample attitude derivative (deg/s) is compared with the measured angular rate in "
                   "deg/s. A disagreement means the two sensors are not telling the same story; it does not "
                   "identify which one is wrong."),
    }


def _rest_mask(a: _Access, canonical_by_header: Dict[str, str], table: pd.DataFrame) -> Optional[np.ndarray]:
    """Samples where the aircraft is evidently at rest: disarmed if possible, else motors at their minimum."""
    armed = a.get("armed")
    if armed is not None and np.isfinite(armed).any() and {0.0, 1.0} <= set(np.unique(armed[np.isfinite(armed)]).tolist()):
        return armed == 0
    motors = [a.get(c) for c in ("motor_1", "motor_2", "motor_3", "motor_4") if a.has(c)]
    if motors:
        mean = np.nanmean(np.vstack(motors), axis=0)
        return mean <= np.nanmin(mean) + 1e-9
    return None


def _limits(timing: Dict[str, Any], rels: Dict[str, Any], canonical_by_header: Dict[str, str],
            specs: Dict[str, ColumnSpec]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    unverified = sorted({specs[c].name for c in canonical_by_header.values()
                         if c in specs and specs[c].unit_status != "verified"})
    if unverified:
        out.append({"topic": "units", "text": "Some channels have no verified unit; their values stay raw."})
    missing = [c for c in ("yaw", "alt_est", "alt_set") if c not in canonical_by_header.values()]
    if missing:
        out.append({"topic": "missing", "text": "Not present in this file: " + ", ".join(missing) + "."})
    out.append({"topic": "conventions",
                "text": "Axis and sign conventions are not established, so no physical direction of motion is stated."})
    out.append({"topic": "cause", "text": "Relationships are measured, not explained: no cause is assigned."})
    return out


# ---- event annotation ---------------------------------------------------------------------------------------------

COMMAND_FOR_RESPONSE = {
    "roll": ("roll_cmd", "roll_tracking"),
    "pitch": ("pitch_cmd", "pitch_tracking"),
    "motor_1": ("throttle_cmd", "throttle_motors"),
    "motor_2": ("throttle_cmd", "throttle_motors"),
    "motor_3": ("throttle_cmd", "throttle_motors"),
    "motor_4": ("throttle_cmd", "throttle_motors"),
    "alt_est": ("alt_set", "altitude_setpoint"),
    "tof_height": ("alt_set", "altitude_setpoint"),
    "tof_mm": ("alt_set", "altitude_setpoint"),
}

# Physical significance floors in the channel's verified unit (or raw counts where the unit is not verified).
# Below the floor a change is numerically detectable but not worth a pilot's attention.
PHYSICAL_FLOOR = {
    "attitude": 0.75,        # deg
    "angular_rate": 5.0,     # deg/s
    "acceleration": 1.0,     # m/s^2
    "altitude": 0.02,        # raw; unit not verified
    "distance": 5.0,         # raw counts
    "motor": 25.0,           # raw output counts
}
AIRCRAFT_CATEGORIES = ("attitude", "acceleration", "angular_rate", "altitude", "distance", "motor")
NON_AIRFRAME_CATEGORIES = ("control", "aux", "status", "command")


def annotate_events(
    events: List[Dict[str, Any]],
    *,
    table: pd.DataFrame,
    specs: Dict[str, ColumnSpec],
    canonical_by_header: Dict[str, str],
    time_s: Optional[np.ndarray],
    relationships: Dict[str, Any],
    window: int,
) -> None:
    """Attach interpretation, command context, sensor consistency and physical values to every event."""
    a = _Access(table, canonical_by_header)

    for ev in events:
        s, e = ev["start_index"], ev["end_index"]
        pre = (max(0, s - window), s)
        cur = (s, e + 1)
        command_checks: List[Dict[str, Any]] = []
        consistency: List[Dict[str, Any]] = []
        physical: List[Dict[str, Any]] = []
        categories: List[str] = []
        max_effect: Dict[str, float] = {}

        for m in ev["measurements"]:
            canonical = canonical_by_header.get(m["channel"])
            spec = specs.get(canonical) if canonical else None
            unit = spec.display_unit if spec else None
            m["canonical"] = canonical
            m["unit"] = unit
            m["unit_status"] = spec.unit_status if spec else "unknown"
            m["signal_category"] = spec.category if spec else None
            if spec:
                categories.append(spec.category)
                effect = max(abs(m.get("shift") or 0.0), abs(m.get("peak_departure") or 0.0))
                max_effect[spec.category] = max(max_effect.get(spec.category, 0.0), effect)
            physical.append({
                "channel": m["channel"], "canonical": canonical, "label": (spec.description if spec and spec.description else (canonical or m["channel"])),
                "category": spec.category if spec else "unknown", "unit": unit, "raw": unit is None,
                "unit_status": spec.unit_status if spec else "unknown",
                "before": _fmt_value(m["before_median"], unit),
                "during": _fmt_value(m["during_median"], unit),
                "range_during": [_fmt_value(m["during_min"], unit), _fmt_value(m["during_max"], unit)],
                "peak": _fmt_value(m.get("peak_departure"), unit),
                "after": _fmt_value(m["after_median"], unit),
                "change": _fmt_value(m["shift"], unit),
            })
            partner = COMMAND_FOR_RESPONSE.get(canonical or "")
            if partner and a.has(partner[0]):
                cc = _command_check(a, canonical or "", partner[0], relationships.get(partner[1]), pre, cur, window, time_s)
                if cc:
                    command_checks.append(cc)
            for rid in _relations_for_channel(canonical, relationships):
                c = _consistency_check(canonical or "", relationships[rid], a, pre, cur)
                if c:
                    consistency.append(c)

        family = ev.get("family")
        if family in ("acceleration",) and "gravity" in relationships:
            consistency.append(_gravity_for_event(a, relationships["gravity"], cur))
        if family in ("altitude", "distance"):
            consistency.extend(_altitude_for_event(relationships, canonical_by_header))
        if family == "motor" and "motor_balance" in relationships:
            consistency.append({"type": "motor_balance", "status": relationships["motor_balance"].get("status"),
                                "detail": relationships["motor_balance"].get("detail")})

        physical_categories = [c for c in categories if c in AIRCRAFT_CATEGORIES]
        significance = "not_applicable"
        if physical_categories:
            if any(max_effect.get(c, 0.0) >= PHYSICAL_FLOOR.get(c, float("inf")) for c in physical_categories):
                significance = "physical"
            else:
                significance = "small"

        ev["physical_values"] = physical
        ev["command_checks"] = command_checks
        ev["consistency"] = _dedupe([c for c in consistency if c])
        ev["signal_categories"] = sorted(set(categories))
        ev["significance"] = significance
        ev["interpretation"] = _interpret(ev, command_checks, ev["consistency"], significance, window)
        ev["interpretation_label"] = INTERPRETATION_LABEL.get(ev["interpretation"], ev["interpretation"])
        if ev["interpretation"] == "control_input_change" and (ev.get("signal_categories") or []) == ["control"]:
            ev["interpretation_label"] = "Controller output change"
        ev["confidence"] = _confidence(ev, command_checks, ev["consistency"], window)
        ev["headline"] = _headline(ev, specs, family)
        ev["expected"] = ev["interpretation"] in ("expected_response", "control_input_change", "status_change")
        if ev["expected"] or significance in ("small", "not_applicable"):
            ev["priority"] = "minor"
        elif ev["interpretation"] in ("unexplained_deviation", "telemetry_inconsistency") and ev["priority"] == "minor":
            ev["priority"] = "review"
        if ev["category"] == "data_quality":
            ev["interpretation"] = "data_quality"
            ev["interpretation_label"] = INTERPRETATION_LABEL["data_quality"]
            ev["expected"] = False


def _command_check(a: _Access, canonical: str, command: str, rel: Optional[Dict[str, Any]],
                   pre: Tuple[int, int], cur: Tuple[int, int], window: int,
                   time_s: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
    """Compare the measured response with the input that could plausibly have caused it.

    The input may have moved before the event (a throttle ramp shows up in motor output later), so a
    bounded look-back is searched for the most recent meaningful step. Direction is learned from the
    whole-recording relationship, never assumed from the sign convention.
    """
    y, c = a.get(canonical), a.get(command)
    if y is None or c is None:
        return None
    pb, pd = c[pre[0]:pre[1]], c[cur[0]:cur[1]]
    yb, yd = y[pre[0]:pre[1]], y[cur[0]:cur[1]]
    pb, pd = pb[np.isfinite(pb)], pd[np.isfinite(pd)]
    yb, yd = yb[np.isfinite(yb)], yd[np.isfinite(yd)]
    if len(pb) < 5 or len(pd) < 3 or len(yb) < 5 or len(yd) < 3:
        return {"command": command, "response": canonical, "status": "insufficient_data"}
    c_delta = float(np.median(pd) - np.median(pb))
    y_delta = float(np.median(yd) - np.median(yb))
    noise = max(_mad(np.diff(c[np.isfinite(c)])), 1e-9) if np.isfinite(c).sum() > 2 else 1e-9
    c_moved = abs(c_delta) > max(4 * noise, 1e-9)
    lag_samples = 0
    if not c_moved:
        step, lag_samples = _recent_step(c, pre[1], lookback=max(window * 3, 30))
        if step is not None:
            c_delta, c_moved = step, True
    sign = (rel or {}).get("direction_sign") or 1
    rel_status = (rel or {}).get("status")
    same_dir = bool(c_moved and y_delta != 0 and math.copysign(1, c_delta * y_delta * sign) > 0)
    if same_dir and rel_status == "insufficient_data":
        status = "insufficient_data"
    elif same_dir and rel_status not in ("consistent", "partially_consistent"):
        status = "relationship_unclear"
    else:
        status = ("explained" if same_dir else
                  "opposite_to_command" if c_moved and y_delta != 0 else
                  "no_command_change" if not c_moved else "unclear")
    return {
        "command": command, "response": canonical,
        "command_change_raw": _r(c_delta), "response_change": _r(y_delta),
        "command_moved": bool(c_moved), "same_direction": same_dir, "status": status,
        "command_lag_s": _r(lag_samples * float(np.median(np.diff(time_s[np.isfinite(time_s)]))), 2)
        if (time_s is not None and lag_samples) else None,
        "relationship_status": (rel or {}).get("status"), "relationship_r": (rel or {}).get("r"),
        "detail": ("The recorded input and the measured response are compared over the event, with a bounded "
                   "look-back for an input change that preceded it. Direction uses the sign learned across the "
                   "recording, not an assumed convention."),
    }


def _recent_step(c: np.ndarray, at: int, lookback: int) -> Tuple[Optional[float], int]:
    """Most recent step in a command series before ``at``: (delta, samples_before_event)."""
    lo = max(0, at - lookback)
    if at - lo < 20:
        return None, 0
    seg = c[lo:at]
    small = max(3, min(10, len(seg) // 6))
    best: Tuple[Optional[float], int] = (None, 0)
    for i in range(small * 2, len(seg) - small + 1):
        before = seg[max(0, i - small * 2): i - small]
        after = seg[i: i + small]
        before, after = before[np.isfinite(before)], after[np.isfinite(after)]
        if len(before) < 2 or len(after) < 2:
            continue
        d = float(np.median(after) - np.median(before))
        if best[0] is None or abs(d) > abs(best[0]):
            best = (d, at - (lo + i))
    if best[0] is not None:
        noise = max(_mad(np.diff(seg[np.isfinite(seg)])), 1e-9) if np.isfinite(seg).sum() > 2 else 1e-9
        if abs(best[0]) < max(4 * noise, 1e-9):
            return None, 0
    return best


def _gravity_for_event(a: _Access, rel: Dict[str, Any], cur: Tuple[int, int]) -> Dict[str, Any]:
    mag = np.sqrt(sum(np.square(a.get(c)) for c in ("acc_x", "acc_y", "acc_z") if a.has(c)))
    seg = mag[cur[0]:cur[1]]
    seg = seg[np.isfinite(seg)]
    peak = float(np.max(seg)) if len(seg) else None
    base = rel.get("rest_median")
    return {"type": "gravity", "status": rel.get("status"), "rest_median": base, "peak_during": _r(peak),
            "detail": "The acceleration magnitude during the event is compared with the at-rest gravity level."}


def _altitude_for_event(relationships: Dict[str, Any], canonical_by_header: Dict[str, str]) -> List[Dict[str, Any]]:
    out = []
    for rid in ("altitude_setpoint", "altitude_independent_measurement"):
        rel = relationships.get(rid)
        if rel:
            out.append({"type": "altitude", "relationship": rid, "status": rel.get("status"),
                        "agreement": rel.get("agreement"), "r": rel.get("r"),
                        "detail": rel.get("label")})
    return out


def _consistency_check(canonical: str, rel: Dict[str, Any], a: _Access,
                       pre: Tuple[int, int], cur: Tuple[int, int]) -> Optional[Dict[str, Any]]:
    kind = rel.get("kind")
    if kind == "physical_consistency" and rel.get("rate") == canonical and rel.get("attitude") != canonical:
        attitude, rate = a.get(rel.get("attitude", "")), a.get(canonical)
        if attitude is None or rate is None:
            return None
        yb, yd = attitude[pre[0]:pre[1]], attitude[cur[0]:cur[1]]
        yb, yd = yb[np.isfinite(yb)], yd[np.isfinite(yd)]
        rd = rate[cur[0]:cur[1]]
        rd = rd[np.isfinite(rd)]
        if not len(rd) or not len(yb) or not len(yd):
            return None
        y_delta, rate_mean = float(np.median(yd) - np.median(yb)), float(np.mean(rd))
        sign = (1 if (rel.get("r") or 0) >= 0 else -1) if rel.get("r") is not None else 1
        if abs(y_delta) < 0.05 and abs(rate_mean) >= 5.0:
            agrees = False   # a large rotation rate with no attitude change: the records disagree
        else:
            agrees = (math.copysign(1, y_delta * rate_mean * sign) > 0) if (y_delta and rate_mean) else None
        return {"type": "attitude_rate", "rate": rel.get("rate"), "status": rel.get("status"),
                "same_direction": agrees, "rate_mean_during": _r(rate_mean), "attitude_change": _r(y_delta),
                "detail": "Attitude change is compared with the measured angular rate over the same interval."}
    if kind == "physical_consistency" and rel.get("attitude") == canonical:
        rate, y = a.get(rel.get("rate", "")), a.get(canonical)
        if rate is None or y is None:
            return None
        rdy = rate[cur[0]:cur[1]]
        rdy = rdy[np.isfinite(rdy)]
        yb, yd = y[pre[0]:pre[1]], y[cur[0]:cur[1]]
        yb, yd = yb[np.isfinite(yb)], yd[np.isfinite(yd)]
        if not len(rdy) or not len(yb) or not len(yd):
            return None
        y_delta = float(np.median(yd) - np.median(yb))
        rate_mean = float(np.mean(rdy))
        sign = (1 if (rel.get("r") or 0) >= 0 else -1) if rel.get("r") is not None else 1
        agrees = (math.copysign(1, y_delta * rate_mean * sign) > 0) if (y_delta and rate_mean) else None
        return {"type": "attitude_rate", "rate": rel.get("rate"), "status": rel.get("status"),
                "same_direction": agrees, "rate_mean_during": _r(rate_mean), "attitude_change": _r(y_delta),
                "detail": "Attitude change is compared with the measured angular rate over the same interval."}
    return None


def _relations_for_channel(canonical: Optional[str], relationships: Dict[str, Any]) -> List[str]:
    if not canonical:
        return []
    out = []
    for rid, rel in relationships.items():
        if (rel.get("response") == canonical or rel.get("attitude") == canonical
                or rel.get("rate") == canonical or rel.get("other") == canonical):
            out.append(rid)
    return out


def _dedupe(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for i in items:
        key = (i.get("type"), i.get("rate"), i.get("attitude"), i.get("relationship"))
        if key in seen:
            continue
        seen.add(key)
        out.append(i)
    return out


def _interpret(ev: Dict[str, Any], commands: List[Dict[str, Any]], consistency: List[Dict[str, Any]],
               significance: str, window: int) -> str:
    if ev["category"] == "data_quality":
        return "data_quality"
    cats = set(ev.get("signal_categories") or [])
    if ev["category"] == "state_transition":
        return "control_input_change" if (cats & {"command", "altitude"}) else "status_change"
    if cats and cats <= set(NON_AIRFRAME_CATEGORIES):
        return "control_input_change"
    has_baseline = any(m.get("before", {}).get("available") for m in ev["measurements"])
    if not has_baseline or window < 10:
        return "insufficient_evidence"
    if significance == "small":
        return "insufficient_evidence"
    if any(c.get("type") == "attitude_rate" and c.get("status") == "not_consistent" and c.get("same_direction") is False
           for c in consistency):
        return "telemetry_inconsistency"
    measured = [m for m in ev["measurements"] if m.get("signal_category") in AIRCRAFT_CATEGORIES]
    explained = [c for c in commands if c.get("status") == "explained"]
    if measured and commands and len(explained) >= max(1, len(commands) // 2):
        return "expected_response"
    return "unexplained_deviation"


def _confidence(ev: Dict[str, Any], commands: List[Dict[str, Any]], consistency: List[Dict[str, Any]], window: int) -> str:
    if window < 20:
        return "low"
    available = sum(1 for m in ev["measurements"] if m.get("before", {}).get("available"))
    if available >= 2 and (any(c.get("status") == "explained" for c in commands) or consistency):
        return "high"
    if available >= 1:
        return "medium"
    return "low"


def _headline(ev: Dict[str, Any], specs: Dict[str, ColumnSpec], family: Optional[str]) -> str:
    interpretation = ev.get("interpretation")
    titles = {
        "attitude": "attitude", "acceleration": "acceleration", "angular_rate": "angular rate",
        "altitude": "altitude", "distance": "height measurement", "motor": "motor output",
        "control": "controller output", "command": "pilot input", "status": "status flag", "aux": "auxiliary channel",
    }
    label = titles.get(family or "", "telemetry")
    if interpretation == "expected_response":
        return f"{label.capitalize()} change consistent with the recorded input"
    if interpretation == "unexplained_deviation":
        return f"{label.capitalize()} change not explained by the recorded input"
    if interpretation == "telemetry_inconsistency":
        return f"Inconsistent {label} readings"
    if interpretation == "control_input_change":
        return "Pilot or command input change"
    if interpretation == "status_change":
        return "Status flag change (meaning not assumed)"
    if interpretation == "data_quality":
        return "Recording problem"
    return f"No significant {label} deviation"


# ---- plain-language rendering of the structured evidence ------------------------------------------------------------

RESPONSE_LABEL = {
    "roll": "roll angle", "pitch": "pitch angle",
    "motor_1": "motor 1 output", "motor_2": "motor 2 output",
    "motor_3": "motor 3 output", "motor_4": "motor 4 output",
    "alt_est": "estimated height", "tof_height": "time-of-flight height", "tof_mm": "time-of-flight distance",
}

COMMAND_LABEL = {
    "roll_cmd": "roll input", "pitch_cmd": "pitch input", "throttle_cmd": "throttle input",
    "yaw_cmd": "yaw input", "alt_set": "altitude setpoint",
}


def physical_summary(entry: Dict[str, Any]) -> str:
    """'Roll 0.1 deg before, -0.66 deg during (range -2.23 to 0.13 deg).' / raw form when no unit is verified."""
    name = entry.get("label") or entry.get("canonical") or entry.get("channel")
    unit = entry.get("unit")
    parts = []
    if entry.get("before") is not None:
        parts.append(f"{entry['before']} before")
    if entry.get("during") is not None:
        parts.append(f"{entry['during']} during")
    if entry.get("peak") is not None:
        parts.append(f"largest departure {entry['peak']}")
    text = f"{name}: " + ", ".join(parts) if parts else f"{name}: values unavailable"
    if entry.get("raw"):
        text += " (raw; unit not verified)"
    elif unit:
        rng = entry.get("range_during") or []
        if len(rng) == 2 and rng[0] is not None and rng[1] is not None:
            text += f" (range {rng[0]} to {rng[1]})"
    return text + "."


def command_sentence(check: Dict[str, Any]) -> str:
    cmd = COMMAND_LABEL.get(str(check.get("command")), str(check.get("command")))
    resp = RESPONSE_LABEL.get(str(check.get("response")), str(check.get("response")))
    status = check.get("status")
    if status == "explained":
        lag = check.get("command_lag_s")
        lead = f"The {cmd} changed"
        if lag:
            lead += f" about {lag} s before this period"
        else:
            lead += " over this period"
        return f"{lead}, and the {resp} change is in the same direction as that input across the recording."
    if status == "no_command_change":
        return f"The {cmd} did not change around this period, so the {resp} change is not explained by it."
    if status == "opposite_to_command":
        return f"The {cmd} changed in the opposite direction to the usual command-response relationship."
    if status == "relationship_unclear":
        return (f"The {cmd} moved, but across the recording it does not track the {resp} closely enough to call "
                f"this a response to it.")
    if status == "insufficient_data":
        return f"Not enough {cmd} variation to compare with the {resp}."
    return f"The {cmd} and {resp} could not be compared cleanly."


def consistency_sentence(item: Dict[str, Any]) -> str:
    typ = item.get("type")
    if typ == "attitude_rate":
        axis = str(item.get("rate") or "rotation rate").replace("gyro_", "").upper()
        if item.get("same_direction") is True:
            return f"The attitude change and the measured rotation rate about {axis} agree in direction over this period."
        if item.get("same_direction") is False:
            return (f"The attitude change and the measured rotation rate about {axis} do not agree in direction "
                    "over this period; the two records are inconsistent.")
        return f"The attitude change and the rotation rate about {axis} could not be compared over this period."
    if typ == "gravity":
        rest = item.get("rest_median")
        peak = item.get("peak_during")
        if rest is not None and peak is not None:
            return (f"At rest the acceleration magnitude was {float(rest):.2f} m/s^2 (about gravity); it reached "
                    f"{float(peak):.2f} m/s^2 during this period.")
        return "The acceleration magnitude could not be compared with gravity over this period."
    if typ == "altitude":
        agree = item.get("agreement")
        if item.get("relationship") == "altitude_setpoint":
            if agree == "close":
                return "The altitude estimate follows the altitude setpoint over this period."
            if agree == "different levels":
                return "The altitude estimate and the setpoint do not sit at the same level over this period."
            return "The altitude estimate and setpoint could not be compared over this period."
        if agree == "close":
            return "Estimated altitude and the time-of-flight height stayed close over this period."
        if agree == "different levels":
            return "Estimated altitude and the time-of-flight height do not sit at the same level over this period."
        return "The altitude sources could not be compared over this period."
    if typ == "motor_balance":
        return "The four motor outputs were compared with each other over the whole recording."
    return ""
