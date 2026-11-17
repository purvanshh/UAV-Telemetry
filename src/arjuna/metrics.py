"""A fixed set of comparable measurements for every flight.

Fix check and flight comparison need the same numbers from two flights, computed the same way. They are computed here,
by code, from the log and the analysis, and stored with the analysis. A measurement the flight does not have is simply
absent (never zero).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from src.arjuna.ulog import ArjunaFlight

# key -> (plain label, unit, better direction, smallest change worth reporting)
SPEC: Dict[str, Dict[str, Any]] = {
    "vibration_p95": {"label": "Shaking (typical level)", "unit": "m/s²", "better": "lower", "min_change": 1.0},
    "vibration_max": {"label": "Shaking (worst moment)", "unit": "m/s²", "better": "lower", "min_change": 2.0},
    "clip_rate_max": {"label": "Motion sensor overload (peak)", "unit": "per second", "better": "lower", "min_change": 2.0},
    "tracking_error_rms": {"label": "Holding its angle (average error)", "unit": "°", "better": "lower", "min_change": 0.3},
    "tracking_error_p99": {"label": "Holding its angle (worst moments)", "unit": "°", "better": "lower", "min_change": 1.0},
    "motor_spread_pct": {"label": "Motor balance in hover (worst motor vs the average)", "unit": "%", "better": "lower", "min_change": 1.5},
    "battery_min_cell_v": {"label": "Lowest battery voltage per cell", "unit": "V", "better": "higher", "min_change": 0.05},
    "battery_sag_cell_v": {"label": "Battery voltage drop under load", "unit": "V", "better": "lower", "min_change": 0.05},
    "gps_min_sats": {"label": "Fewest GPS satellites", "unit": "", "better": "higher", "min_change": 1.0},
    "gps_max_hacc_m": {"label": "Worst GPS position error", "unit": "m", "better": "lower", "min_change": 0.3},
    "tilt_max_deg": {"label": "Steepest tilt", "unit": "°", "better": "lower", "min_change": 3.0},
    "dropouts": {"label": "Gaps in the recording", "unit": "", "better": "lower", "min_change": 1.0},
}


def _r(x: Any, nd: int = 3) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def key_metrics(fl: ArjunaFlight, a: Dict[str, Any]) -> Dict[str, Any]:
    """Measurements over the time in the air (a log that is not a flight has none except the recording gaps)."""
    out: Dict[str, Optional[float]] = {"dropouts": float(len(fl.dropouts))}
    if (a.get("classification") or {}).get("kind") != "flight":
        return {k: out[k] for k in out}
    m = a.get("metrics") or {}
    out["tracking_error_rms"] = _r(m.get("att_err_rms_deg"), 2)
    out["tracking_error_p99"] = _r(m.get("att_err_p99_deg"), 2)
    pw, gp = m.get("power") or {}, m.get("gps") or {}
    out["battery_min_cell_v"] = _r(pw.get("min_cell_v"), 2)
    out["battery_sag_cell_v"] = _r(pw.get("sag_cell_v"), 2)
    out["gps_min_sats"] = _r(gp.get("min_sats"), 0)
    out["gps_max_hacc_m"] = _r(gp.get("max_hacc_m"), 2)
    hb = (a.get("physics") or {}).get("hover_balance") or {}
    if hb.get("status") == "computed":
        out["motor_spread_pct"] = _r(abs(hb.get("worst_deviation_pct") or 0), 1)

    vb = fl.get("arjuna_vibration")
    if vb is not None:
        air = fl.in_air(vb["t"].to_numpy(float))
        axes = [vb[f"accel_vibe[{i}]"].to_numpy(float)[air] for i in range(3) if f"accel_vibe[{i}]" in vb]
        axes = [x[np.isfinite(x)] for x in axes]
        axes = [x for x in axes if len(x)]
        if axes:
            out["vibration_p95"] = _r(max(np.percentile(x, 95) for x in axes), 1)
            out["vibration_max"] = _r(max(x.max() for x in axes), 1)
        if "clip_rate" in vb:
            c = vb["clip_rate"].to_numpy(float)[air]
            c = c[np.isfinite(c)]
            if len(c):
                out["clip_rate_max"] = _r(c.max(), 0)
    at = fl.get("arjuna_attitude")
    if at is not None:
        air = fl.in_air(at["t"].to_numpy(float))
        if air.any():
            tilt = np.degrees(np.maximum(np.abs(at["roll"].to_numpy(float)), np.abs(at["pitch"].to_numpy(float))))[air]
            if np.isfinite(tilt).any():
                out["tilt_max_deg"] = _r(np.nanmax(tilt), 0)
    return {k: v for k, v in out.items() if v is not None}
