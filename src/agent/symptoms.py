"""The pilot's "what I saw" report as a search query over the flight (Roadmap, "Linking 'what I saw' to the log").

A symptom from a fixed list plus a rough time. The time becomes a +-30 s search window on the log's own axis using
take-off and landing as anchors. Every finding on the whole flight is then ranked by how well its fault code fits
the symptom (fixed prior table), how close it is in time, and the detector's confidence. Pure code: the agent reads
the ranking, it does not make it.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

SEARCH_HALF_WIDTH_S = 30.0

SYMPTOMS: Dict[str, Dict[str, Any]] = {
    "wobble": {"label": "Wobble or oscillation", "codes": ["CTRL_OSC", "MOTOR_ESC", "VIB_HIGH", "CTRL_SAT"]},
    "drift": {"label": "Drift (would not hold position)", "codes": ["GPS_DEGRADED", "EST_DIVERGE", "SENSOR_FAIL", "COMMS"]},
    "sudden_drop": {"label": "Sudden drop in height", "codes": ["POWER", "MOTOR_ESC", "CTRL_SAT", "CRASH"]},
    "yaw_spin": {"label": "Yaw spin", "codes": ["MOTOR_ESC", "CTRL_SAT", "SENSOR_FAIL", "EST_DIVERGE"]},
    "toilet_bowl": {"label": "Toilet-bowling (circling)", "codes": ["SENSOR_FAIL", "EST_DIVERGE", "GPS_DEGRADED"]},
    "slow_response": {"label": "Slow or weak response to sticks", "codes": ["CTRL_SAT", "POWER", "TIMING", "CTRL_OSC"]},
    "failsafe": {"label": "Failsafe triggered", "codes": ["COMMS", "POWER", "GPS_DEGRADED", "EST_DIVERGE", "VIB_HIGH"]},
    "crash": {"label": "Crash", "codes": ["CRASH", "MOTOR_ESC", "POWER", "CTRL_SAT", "SENSOR_FAIL", "VIB_HIGH"]},
    "motor_sound": {"label": "Motor sound or smell", "codes": ["MOTOR_ESC", "VIB_HIGH", "POWER"]},
    "other": {"label": "Something else", "codes": []},
}
PRIOR_STEPS = [1.0, 0.85, 0.7, 0.6, 0.5, 0.45]
PRIOR_UNLISTED = 0.2
CONFIDENCE_WEIGHT = {"high": 1.0, "medium": 0.75, "low": 0.5}
TIME_REFS = {"since_takeoff": "seconds after take-off", "since_log_start": "seconds after the log started",
             "before_landing": "seconds before landing", "phase": "a flight phase"}
PHASES = {"takeoff", "climb", "cruise", "landing"}


def search_window(report: Dict[str, Any], takeoff: Optional[float], landing: Optional[float],
                  duration: float) -> Dict[str, Any]:
    ref, val = report.get("time_ref"), report.get("time_value")
    centre: Optional[float] = None
    half = SEARCH_HALF_WIDTH_S
    note = ""
    if ref == "since_log_start" and isinstance(val, (int, float)):
        centre = float(val)
    elif ref == "since_takeoff" and isinstance(val, (int, float)):
        centre = (takeoff or 0.0) + float(val)
        note = "" if takeoff is not None else "no take-off found, so the log start was used"
    elif ref == "before_landing" and isinstance(val, (int, float)):
        centre = (landing if landing is not None else duration) - float(val)
    elif ref == "phase" and val in PHASES:
        a, b = takeoff or 0.0, landing if landing is not None else duration
        if val == "takeoff":
            return {"t_start": a - 10, "t_end": a + 30, "basis": "take-off phase"}
        if val == "landing":
            return {"t_start": b - 40, "t_end": b + 5, "basis": "landing phase"}
        if val == "climb":
            return {"t_start": a, "t_end": a + 60, "basis": "first minute after take-off"}
        return {"t_start": a + 30, "t_end": b - 30, "basis": "between take-off and landing"}
    if centre is None:
        return {"t_start": 0.0, "t_end": duration, "basis": "no usable time given; whole flight"}
    return {"t_start": max(0.0, centre - half), "t_end": centre + half, "t_reported": round(centre, 1),
            "basis": f"{val} {TIME_REFS.get(ref, ref)}, +-{half:.0f} s" + (f" ({note})" if note else "")}


def _distance(a0: float, a1: float, b0: float, b1: float) -> float:
    if a1 >= b0 and b1 >= a0:
        return 0.0
    return min(abs(a0 - b1), abs(b0 - a1))


def rank(report: Dict[str, Any], findings: List[Dict[str, Any]], takeoff: Optional[float], landing: Optional[float],
         duration: float) -> Dict[str, Any]:
    sym = SYMPTOMS.get(report.get("symptom") or "other", SYMPTOMS["other"])
    win = search_window(report, takeoff, landing, duration)
    ranked = []
    for f in findings:
        code = f.get("code")
        if sym["codes"]:
            prior = PRIOR_STEPS[min(sym["codes"].index(code), len(PRIOR_STEPS) - 1)] if code in sym["codes"] else PRIOR_UNLISTED
        else:
            prior = 0.5
        s, e = f.get("t_start"), f.get("t_end")
        if s is None or e is None:
            prox = 0.5
            dist = None
        else:
            dist = _distance(float(s), float(e), win["t_start"], win["t_end"])
            # A cause often starts before the symptom: findings that begin up to 60 s earlier are not penalised as much.
            early = float(s) < win["t_start"]
            prox = math.exp(-0.5 * (dist / (60.0 if early else SEARCH_HALF_WIDTH_S)) ** 2)
        conf = CONFIDENCE_WEIGHT.get(f.get("confidence", "high"), 0.7)
        ranked.append({"finding_id": f.get("id"), "code": code, "title": f.get("title"), "when": f.get("when"),
                       "symptom_fit": round(prior, 2), "time_proximity": round(prox, 2), "confidence_weight": conf,
                       "seconds_outside_window": None if dist is None else round(dist, 1),
                       "score": round(prior * prox * conf, 3)})
    ranked.sort(key=lambda r: -r["score"])
    return {"symptom": report.get("symptom"), "symptom_label": sym["label"], "note": report.get("note") or "",
            "search_window": {k: (round(v, 1) if isinstance(v, float) else v) for k, v in win.items()},
            "expected_codes": sym["codes"], "ranking": ranked,
            "method": "score = symptom fit (fixed prior table) x time proximity (Gaussian, 30 s; 60 s for findings that start "
                      "earlier) x detector confidence"}
