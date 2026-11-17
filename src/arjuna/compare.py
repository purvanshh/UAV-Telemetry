"""Fix check and flight comparison: two analyses of the same kind of data, in plain words.

``before`` is the earlier flight, ``after`` the later one. The result says, per kind of problem, whether it is fixed,
still there, or new, and shows the measured change in plain numbers. Everything here is computed by code from the two
stored analyses; nothing is guessed, and a measurement missing from either flight is not compared.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.arjuna.metrics import SPEC
from src.arjuna.pilot import PLAIN

REL_CHANGE = 0.10  # a change must also be at least 10% of the earlier value to count
SEV = {"critical": 2, "warning": 1}

# The measurements that show whether a given kind of problem got better (first = the one most often behind it).
CODE_METRICS = {"VIB_HIGH": ["clip_rate_max", "vibration_p95"], "CTRL_OSC": ["tracking_error_rms", "tracking_error_p99"],
                "MOTOR_ESC": ["motor_spread_pct"], "POWER": ["battery_min_cell_v", "battery_sag_cell_v"],
                "GPS_DEGRADED": ["gps_min_sats", "gps_max_hacc_m"], "CRASH": ["tilt_max_deg"], "LOGGING": ["dropouts"]}


def _num(v: float, unit: str) -> str:
    s = f"{v:.0f}" if abs(v) >= 100 else f"{v:.2f}".rstrip("0").rstrip(".")
    return f"{s}{unit}" if unit in ("°", "%") else f"{s} {unit}".strip()


def _evidence(mm: Dict[str, Dict[str, Any]], code: str, state: str) -> Optional[str]:
    """The measurement line that explains this problem's outcome; never one that contradicts it."""
    rows = [mm[k] for k in CODE_METRICS.get(code, []) if k in mm]
    if not rows:
        return None
    want = {"fixed": "better", "better": "better", "worse": "worse", "new": "worse", "new_unchecked": "worse"}.get(state)
    pick = next((r for r in rows if r["status"] == want), None) or next((r for r in rows if r["status"] == "same"), None)
    if pick is None and state in ("fixed", "better"):
        pick = rows[0]
        return pick["text"] + " (a little higher than before, but within its limit)" if state == "fixed" else None
    return pick["text"] if pick else None


def _metric_rows(a: Dict[str, Any], b: Dict[str, Any]) -> List[Dict[str, Any]]:
    ma, mb = a.get("key_metrics") or {}, b.get("key_metrics") or {}
    rows = []
    for key, spec in SPEC.items():
        if key not in ma or key not in mb:
            continue
        va, vb = float(ma[key]), float(mb[key])
        delta = vb - va
        big = abs(delta) >= spec["min_change"] and abs(delta) >= REL_CHANGE * max(abs(va), 1e-9)
        if not big:
            status = "same"
        else:
            status = "better" if (delta < 0) == (spec["better"] == "lower") else "worse"
        pct = (delta / abs(va) * 100) if va else None
        text = f"{spec['label']}: {_num(va, spec['unit'])} → {_num(vb, spec['unit'])}"
        if status != "same" and pct is not None:
            text += f" ({'down' if delta < 0 else 'up'} {abs(pct):.0f}%)"
        rows.append({"key": key, "label": spec["label"], "unit": spec["unit"], "before": va, "after": vb,
                     "change_pct": round(pct, 1) if pct is not None else None, "status": status, "text": text})
    return rows


def _by_code(summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for f in summary.get("findings") or []:
        cur = out.get(f["code"])
        if cur is None or SEV[f["severity"]] > SEV[cur["severity"]]:
            out[f["code"]] = f
    return out


def _evaluated(summary: Dict[str, Any], code: str) -> bool:
    return ((summary.get("coverage") or {}).get(code) or {}).get("status") in ("evaluated", "partial")


def _comparable(a: Dict[str, Any], b: Dict[str, Any]) -> Optional[str]:
    for name, s in (("The earlier", a), ("The later", b)):
        if s.get("refused"):
            return f"{name} log could not be read, so the two flights cannot be compared."
        if (s.get("classification") or {}).get("kind") != "flight":
            return f"{name} recording is not a full flight (a ground or bench test), so the flights cannot be compared."
    return None


def compare(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    why = _comparable(before, after)
    if why:
        return {"status": "not_comparable", "status_text": "These two cannot be compared", "status_detail": why,
                "problems": [], "metrics": [], "notes": []}
    notes: List[str] = []
    ta, tb = (before.get("timeline") or {}).get("airborne_s"), (after.get("timeline") or {}).get("airborne_s")
    if ta and tb and (max(ta, tb) / max(min(ta, tb), 1e-9)) > 3:
        notes.append("The flights are very different in length, so the comparison is only a rough guide.")
    sa, sb = (before.get("conversion") or {}).get("format", {}).get("id"), (after.get("conversion") or {}).get("format", {}).get("id")
    if sa != sb:
        notes.append("The two logs come from different sources, so some measurements may not be comparable.")

    metrics = _metric_rows(before, after)
    mm = {m["key"]: m for m in metrics}
    cb, ca = _by_code(before), _by_code(after)
    problems: List[Dict[str, Any]] = []
    for code, f in cb.items():
        name = PLAIN.get(code, {}).get("name", code)
        rows = [mm[k] for k in CODE_METRICS.get(code, []) if k in mm]
        got_better = any(r["status"] == "better" for r in rows) and not any(r["status"] == "worse" for r in rows)
        got_worse = any(r["status"] == "worse" for r in rows) and not any(r["status"] == "better" for r in rows)
        if code in ca:
            g = ca[code]
            if SEV[g["severity"]] < SEV[f["severity"]] or got_better:
                state, text = "better", "Still there, but less severe."
            elif SEV[g["severity"]] > SEV[f["severity"]] or got_worse:
                state, text = "worse", "Still there, and worse."
            else:
                state, text = "still", "Still there."
        elif _evaluated(after, code):
            state, text = "fixed", "Fixed. It did not show up on the later flight."
        else:
            state, text = "unknown", "The later flight did not record what is needed to check this."
        problems.append({"code": code, "name": name, "state": state, "text": text, "evidence": _evidence(mm, code, state),
                         "before_severity": f["severity"], "after_severity": ca[code]["severity"] if code in ca else None})
    new = [code for code in ca if code not in cb]
    for code in new:
        name = PLAIN.get(code, {}).get("name", code)
        seen = _evaluated(before, code)
        problems.append({"code": code, "name": name, "state": "new" if seen else "new_unchecked",
                         "text": "New: this was not found on the earlier flight." if seen else
                                 "Found now. The earlier flight did not record what is needed to check it.",
                         "evidence": _evidence(mm, code, "new"), "before_severity": None,
                         "after_severity": ca[code]["severity"]})

    states = [p["state"] for p in problems if p["state"] != "unknown"]
    fixed = sum(s == "fixed" for s in states)
    open_ = sum(s in ("still", "worse", "better") for s in states)
    new_n = sum(s in ("new", "new_unchecked") for s in states)
    new_critical = any(p["state"] in ("new", "new_unchecked") and p["after_severity"] == "critical" for p in problems)
    worse = any(p["state"] == "worse" for p in problems)
    if not cb and not ca:
        status, text = "no_problems", "No problems in either flight"
        detail = "Neither flight had a problem on the checks that could run."
    elif not cb:
        status, text = "worse", "New problems on the later flight"
        detail = "The earlier flight had no problems, but the later one does."
    elif new_critical or worse:
        status, text = "worse", "Got worse"
        detail = "The later flight is worse than the earlier one."
    elif fixed and not open_ and not new_n:
        status, text = "fixed", "Fixed"
        detail = "Every problem from the earlier flight is gone."
    elif fixed:
        status, text = "partly_fixed", "Partly fixed"
        detail = f"{fixed} problem(s) gone"
        detail += f", {open_} still there" if open_ else ""
        detail += f", {new_n} new" if new_n else ""
        detail += "."
    elif any(p["state"] == "better" for p in problems) and not new_n:
        status, text = "partly_fixed", "Better, but not fixed"
        detail = "The problem is less severe but still there."
    else:
        status, text = "not_fixed", "Not fixed"
        detail = "The problem from the earlier flight is still there."
    order = {"worse": 0, "new": 1, "new_unchecked": 2, "still": 3, "better": 4, "unknown": 5, "fixed": 6}
    problems.sort(key=lambda p: order[p["state"]])
    return {"status": status, "status_text": text, "status_detail": detail, "problems": problems, "metrics": metrics,
            "notes": notes}
