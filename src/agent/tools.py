"""The agent's tools. Each one is plain numpy over the flight; the agent chooses what to call and with which arguments.

Every call gets a reference (T1, T2, ...). The report must cite these references, and every number it writes is
checked against the numbers these tools returned (see ``validate.py``). Tools never return position coordinates.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.agent import symptoms
from src.agent.sources import Source, SignalError
from src.arjuna.checks import FAULT_CODES, TH as _TH, clock

MAX_RESULT_CHARS = 14_000
MAX_INTERVALS = 25


def _r(x: Any, nd: int = 4) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return round(v, nd) if abs(v) < 1e6 else round(v)


# ---- JSON schemas ---------------------------------------------------------------------------------------------------

def _fn(name: str, desc: str, props: Dict[str, Any], required: Optional[List[str]] = None) -> Dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": required or [], "additionalProperties": False}}}


SIG = {"topic": {"type": "string", "description": "Topic name, e.g. arjuna_vibration."},
       "field": {"type": "string", "description": "Field name exactly as listed by list_signals, e.g. accel_vibe[2]."},
       "instance": {"type": "integer", "description": "Topic instance (multi_id). Default 0."}}
WIN = {"t_start": {"type": "number", "description": "Window start, seconds since log start. Omit for the whole log."},
       "t_end": {"type": "number", "description": "Window end, seconds since log start. Omit for the whole log."}}
AIR = {"air_only": {"type": "boolean", "description": "Only samples while airborne (default true for flight logs)."}}

COMMON_TOOLS = [
    _fn("get_flight_overview", "Identity, integrity, compatibility, flight classification, timeline (air time, modes, "
        "events, warnings, dropouts), topics present and missing, and which fault codes could be evaluated.", {}),
    _fn("get_findings", "Findings produced by the code detectors (rule checks and physics), with fault code, severity, "
        "time window, detector, confidence and evidence. Optionally filter by fault code.",
        {"code": {"type": "string", "description": "Optional fault code filter, e.g. MOTOR_ESC."}}),
    _fn("list_signals", "Topics, instances, fields, units and logging rates available in this log.",
        {"topic": {"type": "string", "description": "Optional topic to list."}}),
    _fn("signal_stats", "Statistics of one signal over a window: n, min, max (with times), mean, median, std, p05, p95, "
        "RMS, linear trend per second, and fraction missing. Angles are in degrees.", {**SIG, **WIN, **AIR}, ["topic", "field"]),
    _fn("compare_periods", "Compare one signal between two windows: statistics of each, and the difference, ratio and "
        "percent change of the medians and of the RMS. Use this instead of doing arithmetic yourself.",
        {**SIG, "a_start": {"type": "number"}, "a_end": {"type": "number"}, "b_start": {"type": "number"},
         "b_end": {"type": "number"}}, ["topic", "field", "a_start", "a_end", "b_start", "b_end"]),
    _fn("find_periods", "Find the time intervals where a signal meets a condition, merged across short gaps. "
        "Conditions: 'above', 'below', 'abs_above', or 'robust_z_above' (distance from the median in robust sigmas).",
        {**SIG, "condition": {"type": "string", "enum": ["above", "below", "abs_above", "robust_z_above"]},
         "value": {"type": "number"}, "min_duration_s": {"type": "number", "description": "Default 0.5"}, **WIN, **AIR},
        ["topic", "field", "condition", "value"]),
    _fn("dominant_frequency", "Spectrum of one signal in a window: strongest frequencies and the share of variance in "
        "each, with the Nyquist limit of the logging rate. Use for oscillation and vibration questions.",
        {**SIG, **WIN}, ["topic", "field"]),
    _fn("correlate", "Pearson correlation of two signals in a window, at zero lag and at the best lag within +-max_lag_s "
        "(positive lag = b follows a).",
        {"a_topic": {"type": "string"}, "a_field": {"type": "string"}, "a_instance": {"type": "integer"},
         "b_topic": {"type": "string"}, "b_field": {"type": "string"}, "b_instance": {"type": "integer"},
         "max_lag_s": {"type": "number"}, **WIN}, ["a_topic", "a_field", "b_topic", "b_field"]),
]

ULOG_TOOLS = [
    _fn("get_physics", "Physics checks: hover motor balance (per-motor command, deviation, imbalance pattern), vertical "
        "thrust residual (predicted vs measured specific force, with sustained departures), and the rotational dynamics "
        "fit with per-motor effectiveness eta (or why it cannot be computed at this logging rate).", {}),
    _fn("controller_tracking", "How one control loop tracked its setpoint in a window: error RMS / max / p95, setpoint "
        "and measured ranges, RMS of P, I, D and FF terms, fraction of time saturated, integrator change. Angles in degrees.",
        {"controller": {"type": "string", "enum": ["rate", "attitude", "altitude", "velocity", "position"]},
         "axis": {"type": "string", "enum": ["roll", "pitch", "yaw", "x", "y", "z"]}, **WIN}, ["controller", "axis"]),
    _fn("motor_outputs", "Per-motor command and PWM in a window: mean, max, time at maximum, spread between motors, "
        "mixer saturation time, and collective thrust demand.", {**WIN}),
    _fn("get_messages", "Logged text messages and flight events in a window, filtered by minimum severity.",
        {**WIN, "min_level": {"type": "string", "enum": ["debug", "info", "notice", "warning", "error", "critical"]}}),
    _fn("sequence_of_departures", "WHAT CHANGED FIRST. For a window (e.g. the 30-60 s before a crash or a critical "
        "finding), compares ~30 key signals (vibration, clipping, attitude, rates, tracking errors, each motor command, "
        "motor spread, thrust demand, battery, altitude, climb rate, GPS, estimator, pilot sticks) with their own "
        "baseline just before the window, and lists when each first departed (robust z > 4 for at least 0.3 s), in "
        "time order, with baseline, onset and peak values. Use it to order the cause chain.",
        {"t_start": {"type": "number"}, "t_end": {"type": "number"},
         "baseline_s": {"type": "number", "description": "Length of the baseline before t_start. Default 60."}},
        ["t_start", "t_end"]),
]

ULOG_TOOLS.append(_fn("state_at", "Snapshot of the vehicle at up to 6 moments: mode, armed, landed state, height and climb "
    "rate, baro altitude, attitude, body rates, thrust demand, each motor command, pilot sticks, battery. Use it to "
    "see the situation at an event (e.g. height near 0 m means the vehicle was on or near the ground).",
    {"times": {"type": "array", "items": {"type": "number"}, "description": "Seconds since log start (max 6)."}}, ["times"]))

# Approved limits shown next to the numbers, so a value is never called "high" when it is far below its limit.
LIMITS = {
    ("arjuna_vibration", "accel_vibe"): {"warning": f"p95 above {_TH['vibe_warn']}", "critical": f"above {_TH['vibe_crit']}", "unit": "m/s^2"},
    ("arjuna_vibration", "clip_rate"): {"warning": f"above {_TH['clip_rate_warn']}", "unit": "1/s"},
    ("arjuna_gps", "satellites"): {"warning": f"below {_TH['gps_sats_warn']}", "unit": "count"},
    ("arjuna_gps", "h_acc"): {"warning": f"above {_TH['gps_hacc_warn']}", "unit": "m"},
    ("arjuna_estimator_innov", "test_ratio"): {"warning": f"above 1 for more than {_TH['innov_ratio_frac_warn'] * 100:.0f}% of air time"},
    ("arjuna_attitude", "roll"): {"critical": f"tilt above {_TH['tilt_crit_deg']}", "unit": "deg"},
    ("arjuna_attitude", "pitch"): {"critical": f"tilt above {_TH['tilt_crit_deg']}", "unit": "deg"},
}


def _limits(topic: str, field: str) -> Optional[Dict[str, Any]]:
    return LIMITS.get((topic, field.split("[", 1)[0]))


STATIC_TOOLS = {"get_flight_overview", "get_findings", "get_physics", "match_pilot_report"}
DEPARTURE_Z = 4.0
DEPARTURE_MIN_S = 0.3

PILOT_TOOL = _fn("match_pilot_report", "Rank every finding against the pilot's reported symptom and rough time "
                 "(fixed symptom-to-fault prior table x time proximity x detector confidence).", {})

SUBMIT = {"type": "function", "function": {
    "name": "submit_report", "strict": True,
    "description": "Submit the final report. Call exactly once, when the investigation is complete.",
    "parameters": {"type": "object", "additionalProperties": False, "required": [
        "headline", "summary", "finding_reviews", "root_cause_chain", "additional_observations", "pilot_report_answer",
        "not_checked", "recommendations"], "properties": {
        "headline": {"type": "string", "description": "One sentence: the most important thing about this flight."},
        "summary": {"type": "string", "description": "At most five sentences for the pilot and engineer."},
        "finding_reviews": {"type": "array", "items": {"type": "object", "additionalProperties": False,
            "required": ["finding_id", "assessment", "explanation", "evidence_refs", "suggested_cause", "check_next"],
            "properties": {
                "finding_id": {"type": "string"},
                "assessment": {"type": "string", "enum": ["confirmed", "likely_false_alarm", "needs_review"]},
                "explanation": {"type": "string", "description": "What happened, when, which signals; cite numbers from tools."},
                "evidence_refs": {"type": "array", "items": {"type": "string"}, "description": "Tool references (T1...) and finding ids."},
                "suggested_cause": {"type": "string", "description": "Likely cause, worded as a suggestion, or empty."},
                "check_next": {"type": "string", "description": "One concrete inspection or check."}}}},
        "root_cause_chain": {"type": "array", "items": {"type": "object", "additionalProperties": False,
            "required": ["finding_id", "role", "link"], "properties": {
                "finding_id": {"type": "string"},
                "role": {"type": "string", "enum": ["root_cause", "consequence", "independent"]},
                "link": {"type": "string", "description": "How it connects physically to the others, in order of time."}}}},
        "additional_observations": {"type": "array", "items": {"type": "object", "additionalProperties": False,
            "required": ["title", "detail", "evidence_refs", "t_start", "t_end", "suggested_code"], "properties": {
                "title": {"type": "string"}, "detail": {"type": "string"},
                "evidence_refs": {"type": "array", "items": {"type": "string"}},
                "t_start": {"type": ["number", "null"]}, "t_end": {"type": ["number", "null"]},
                "suggested_code": {"type": ["string", "null"], "description": "Fault code this may belong to, or null."}}}},
        "pilot_report_answer": {"type": "string", "description": "Answer to the pilot's report, or empty if none."},
        "not_checked": {"type": "array", "items": {"type": "string"}},
        "recommendations": {"type": "array", "items": {"type": "object", "additionalProperties": False,
            "required": ["action", "reason", "finding_ids"], "properties": {
                "action": {"type": "string"}, "reason": {"type": "string"},
                "finding_ids": {"type": "array", "items": {"type": "string"}}}}}}}}}

CONTROLLER_IDS = {"rate": 0, "attitude": 1, "altitude": 2, "velocity": 3, "position": 4}
AXIS_IDS = {"roll": 0, "pitch": 1, "yaw": 2, "x": 3, "y": 4, "z": 5}
LEVEL_RANK = {"debug": 0, "info": 1, "notice": 2, "warning": 3, "error": 4, "critical": 5, "alert": 6, "emergency": 7}


def _conversion_brief(c: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """What the agent needs to know about a CSV conversion: source format, mapping, derivations and assumptions."""
    if not c:
        return None
    return {"source_format": c.get("format"), "time": c.get("time"),
            "mapped": [f"{m['from']} -> {', '.join(m['to'])}" + (f" ({m['conversion']})" if m.get("conversion") else "")
                       for m in c.get("mapped", [])],
            "derived": c.get("derived"), "dropped": c.get("dropped"), "assumptions": c.get("assumptions"),
            "air_rule": c.get("air_rule"), "motor_layout": c.get("motor_layout")}


class Toolbox:
    def __init__(self, source: Source, analysis: Dict[str, Any], pilot_report: Optional[Dict[str, Any]] = None):
        self.src, self.a, self.pilot = source, analysis, pilot_report
        self.calls: List[Dict[str, Any]] = []
        self.results: Dict[str, Any] = {}
        self._seen: Dict[str, str] = {}
        self._static: Dict[str, str] = {}
        self._fns: Dict[str, Callable[..., Any]] = {
            "get_flight_overview": self.get_flight_overview, "get_findings": self.get_findings,
            "list_signals": self.list_signals, "signal_stats": self.signal_stats, "compare_periods": self.compare_periods,
            "find_periods": self.find_periods, "dominant_frequency": self.dominant_frequency, "correlate": self.correlate,
            "get_physics": self.get_physics, "controller_tracking": self.controller_tracking,
            "motor_outputs": self.motor_outputs, "get_messages": self.get_messages, "match_pilot_report": self.match_pilot_report,
            "sequence_of_departures": self.sequence_of_departures, "state_at": self.state_at,
        }

    # ---- registry -------------------------------------------------------------------------------------------------
    def specs(self) -> List[Dict[str, Any]]:
        tools = list(COMMON_TOOLS) + ULOG_TOOLS
        if self.pilot:
            tools.append(PILOT_TOOL)
        return tools + [SUBMIT]

    def call(self, name: str, args: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        key = name + json.dumps(args, sort_keys=True, default=str)
        if name in STATIC_TOOLS and name in self._static:
            old = self._static[name]
            return old, {"ref": old, "duplicate": True,
                         "note": f"You already have the complete result in {old}; it does not change. Read it there."}
        if key in self._seen:
            old = self._seen[key]
            return old, {"ref": old, "duplicate": True, "note": f"Identical call already made; use result {old}."}
        ref = f"T{len(self.calls) + 1}"
        if name not in self._fns or name not in {t["function"]["name"] for t in self.specs()}:
            out: Dict[str, Any] = {"error": f"Unknown tool {name}."}
        else:
            try:
                out = self._fns[name](**args)
            except SignalError as exc:
                out = {"error": str(exc)}
            except TypeError as exc:
                out = {"error": f"Bad arguments for {name}: {exc}"}
            except Exception as exc:  # a tool bug must not end the investigation
                out = {"error": f"{name} failed: {type(exc).__name__}"}
        out = {"ref": ref, **out}
        text = json.dumps(out, default=str, separators=(",", ":"))
        if len(text) > MAX_RESULT_CHARS:
            out = {"ref": ref, "truncated": True, "note": "Result too long; narrow the window or filter.",
                   "partial": text[: MAX_RESULT_CHARS - 200]}
            text = json.dumps(out, separators=(",", ":"))
        self.results[ref] = out
        self._seen[key] = ref
        if name in STATIC_TOOLS and "error" not in out and not args:
            self._static[name] = ref
        self.calls.append({"ref": ref, "tool": name, "args": args, "ok": "error" not in out, "chars": len(text),
                           "error": out.get("error")})
        return ref, out

    # ---- helpers ----------------------------------------------------------------------------------------------------
    def _window(self, t: np.ndarray, t_start: Optional[float], t_end: Optional[float], air_only: Optional[bool]) -> np.ndarray:
        lo, hi = self.src.bounds()
        a = lo if t_start is None else float(t_start)
        b = hi if t_end is None else float(t_end)
        if b < a:
            raise SignalError(f"Window end {b} is before start {a}.")
        m = (t >= a) & (t <= b)
        if air_only is None or air_only:
            m &= self.src.air_mask(t)
        return m

    @staticmethod
    def _stats(t: np.ndarray, y: np.ndarray, unit: Optional[str]) -> Dict[str, Any]:
        fin = np.isfinite(y)
        out: Dict[str, Any] = {"unit": unit, "n": int(len(y)), "missing_fraction": _r(1 - fin.mean(), 4) if len(y) else None}
        if fin.sum() == 0:
            return {**out, "note": "no valid samples in this window"}
        yf, tf = y[fin], t[fin]
        q = np.percentile(yf, [5, 50, 95])
        i_min, i_max = int(np.argmin(yf)), int(np.argmax(yf))
        slope = float(np.polyfit(tf - tf[0], yf, 1)[0]) if len(yf) > 2 and np.ptp(tf) > 0 else None
        return {**out, "min": _r(yf[i_min]), "t_min": _r(tf[i_min], 2), "at_min": clock(tf[i_min]),
                "max": _r(yf[i_max]), "t_max": _r(tf[i_max], 2), "at_max": clock(tf[i_max]),
                "mean": _r(yf.mean()), "median": _r(q[1]), "std": _r(yf.std()), "p05": _r(q[0]), "p95": _r(q[2]),
                "rms": _r(np.sqrt(np.mean(yf ** 2))), "trend_per_s": _r(slope, 6),
                "t_first": _r(tf[0], 2), "t_last": _r(tf[-1], 2)}

    # ---- tools ------------------------------------------------------------------------------------------------------
    def get_flight_overview(self) -> Dict[str, Any]:
        a = self.a
        tl = a.get("timeline") or {}
        return {
            "identity": a.get("identity"), "integrity": {k: (a.get("integrity") or {}).get(k) for k in
                                                                         ("result", "label", "detail", "converted_log")},
            "compatibility": a.get("compatibility"), "classification": a.get("classification"),
            "time_basis": self.src.time_basis,
            "timeline": {"duration_s": tl.get("duration_s"), "airborne_s": tl.get("airborne_s"), "air_intervals": tl.get("air_intervals"),
                         "modes": (tl.get("modes") or [])[:25], "events": (tl.get("events") or [])[:40],
                         "important_messages": (tl.get("important_messages") or [])[:30],
                         "dropouts": {"count": len(tl.get("dropouts") or []), "first": (tl.get("dropouts") or [])[:5]}},
            "topics_present": [{k: r[k] for k in ("topic", "instance", "rate_hz", "samples")} for r in (a.get("topics") or {}).get("topics", [])],
            "missing_standard_topics": (a.get("topics") or {}).get("missing_standard_topics"),
            "fault_code_coverage": a.get("coverage"), "verdict": a.get("verdict"),
            "conversion": _conversion_brief(a.get("conversion")),
        }

    def get_findings(self, code: Optional[str] = None) -> Dict[str, Any]:
        fs = [f for f in self.a.get("findings", []) if not code or f["code"] == code]
        return {"count": len(fs), "findings": fs, "fault_codes": FAULT_CODES}

    def list_signals(self, topic: Optional[str] = None) -> Dict[str, Any]:
        return {"time_basis": self.src.time_basis, "signals": self.src.signals(topic)}

    def signal_stats(self, topic: str, field: str, instance: int = 0, t_start: Optional[float] = None,
                     t_end: Optional[float] = None, air_only: Optional[bool] = None) -> Dict[str, Any]:
        t, y, unit = self.src.series(topic, field, instance)
        m = self._window(t, t_start, t_end, air_only)
        out = {"signal": f"{topic}[{instance}].{field}", "window": [t_start, t_end],
               "air_only": air_only is not False, **self._stats(t[m], y[m], unit)}
        if _limits(topic, field):
            out["approved_limits"] = _limits(topic, field)
        return out

    def compare_periods(self, topic: str, field: str, a_start: float, a_end: float, b_start: float, b_end: float,
                        instance: int = 0) -> Dict[str, Any]:
        t, y, unit = self.src.series(topic, field, instance)
        A = self._stats(t[(t >= a_start) & (t <= a_end)], y[(t >= a_start) & (t <= a_end)], unit)
        B = self._stats(t[(t >= b_start) & (t <= b_end)], y[(t >= b_start) & (t <= b_end)], unit)
        out = {"signal": f"{topic}[{instance}].{field}", "unit": unit, "a": {"window": [a_start, a_end], **A},
               "b": {"window": [b_start, b_end], **B}}
        if _limits(topic, field):
            out["approved_limits"] = _limits(topic, field)
        for k in ("median", "rms", "max", "std"):
            va, vb = A.get(k), B.get(k)
            if va is not None and vb is not None:
                out[f"{k}_difference_b_minus_a"] = _r(vb - va)
                if va != 0:
                    out[f"{k}_ratio_b_over_a"] = _r(vb / va)
                    out[f"{k}_percent_change"] = _r((vb - va) / abs(va) * 100, 2)
        return out

    def find_periods(self, topic: str, field: str, condition: str, value: float, instance: int = 0,
                     min_duration_s: float = 0.5, t_start: Optional[float] = None, t_end: Optional[float] = None,
                     air_only: Optional[bool] = None) -> Dict[str, Any]:
        t, y, unit = self.src.series(topic, field, instance)
        m = self._window(t, t_start, t_end, air_only)
        if condition == "robust_z_above":
            fin = np.isfinite(y) & m
            med = float(np.median(y[fin])) if fin.any() else 0.0
            sig = float(np.median(np.abs(y[fin] - med)) * 1.4826) if fin.any() else 0.0
            score = np.abs(y - med) / (sig or 1e-12)
            hit = (score > value) & m
            ref = {"median": _r(med), "robust_sigma": _r(sig)}
        else:
            hit = {"above": y > value, "below": y < value, "abs_above": np.abs(y) > value}[condition] & m
            ref = {}
        d = np.diff(t)
        gap = max(1.5 * float(np.median(d[d > 0])) if (d > 0).any() else 0.5, 0.5)
        from src.arjuna.checks import intervals
        ivs = intervals(t, hit & np.isfinite(y), merge_gap=gap, min_len=min_duration_s)
        rows = []
        for a, b in ivs[:MAX_INTERVALS]:
            sel = (t >= a) & (t <= b) & np.isfinite(y)
            ext = y[sel][np.argmax(np.abs(y[sel]))] if sel.any() else np.nan
            rows.append({"t_start": _r(a, 2), "t_end": _r(b, 2), "when": f"{clock(a)}–{clock(b)}", "duration_s": _r(b - a, 2),
                         "extreme": _r(ext), "median": _r(np.median(y[sel])) if sel.any() else None})
        return {"signal": f"{topic}[{instance}].{field}", "unit": unit, "condition": f"{condition} {value}", **ref,
                "intervals_found": len(ivs), "total_duration_s": _r(sum(b - a for a, b in ivs), 2), "intervals": rows}

    def dominant_frequency(self, topic: str, field: str, instance: int = 0, t_start: Optional[float] = None,
                           t_end: Optional[float] = None) -> Dict[str, Any]:
        t, y, unit = self.src.series(topic, field, instance)
        m = self._window(t, t_start, t_end, False) & np.isfinite(y)
        if m.sum() < 64:
            return {"error": f"Only {int(m.sum())} samples in the window; need at least 64 for a spectrum."}
        tt, yy = t[m], y[m]
        dt = float(np.median(np.diff(tt)))
        if dt <= 0:
            return {"error": "Time does not advance in this window (CSV without a time column?)."}
        grid = np.arange(tt[0], tt[-1], dt)
        yi = np.interp(grid, tt, yy)
        yi = (yi - yi.mean()) * np.hanning(len(yi))
        p = np.abs(np.fft.rfft(yi)) ** 2
        f = np.fft.rfftfreq(len(yi), dt)
        p[0] = 0
        total = p.sum() or 1.0
        top = np.argsort(p)[::-1][:5]
        return {"signal": f"{topic}[{instance}].{field}", "sample_rate_hz": _r(1 / dt, 2), "nyquist_hz": _r(0.5 / dt, 2),
                "resolution_hz": _r(f[1] if len(f) > 1 else None, 4), "samples": int(len(yi)),
                "peaks": [{"frequency_hz": _r(f[i], 3), "share_of_variance": _r(p[i] / total, 4)} for i in top if f[i] > 0],
                "note": "Frequencies above the Nyquist limit cannot be seen and may alias into this range."}

    def correlate(self, a_topic: str, a_field: str, b_topic: str, b_field: str, a_instance: int = 0, b_instance: int = 0,
                  max_lag_s: float = 2.0, t_start: Optional[float] = None, t_end: Optional[float] = None) -> Dict[str, Any]:
        ta, ya, ua = self.src.series(a_topic, a_field, a_instance)
        tb, yb, ub = self.src.series(b_topic, b_field, b_instance)
        lo, hi = self.src.bounds()
        a0 = max(ta[0], tb[0], lo if t_start is None else t_start)
        a1 = min(ta[-1], tb[-1], hi if t_end is None else t_end)
        dt = max(float(np.median(np.diff(ta))), float(np.median(np.diff(tb))), 1e-3)
        grid = np.arange(a0, a1, dt)
        if len(grid) < 20:
            return {"error": "Fewer than 20 common samples in the window."}
        ok_a, ok_b = np.isfinite(ya), np.isfinite(yb)
        A = np.interp(grid, ta[ok_a], ya[ok_a])
        Bv = np.interp(grid, tb[ok_b], yb[ok_b])
        if np.std(A) == 0 or np.std(Bv) == 0:
            return {"error": "One signal is constant in the window."}
        r0 = float(np.corrcoef(A, Bv)[0, 1])
        best, lag = r0, 0
        for k in range(1, int(max_lag_s / dt) + 1):
            for s in (k, -k):
                x, y2 = (A[:-s], Bv[s:]) if s > 0 else (A[-s:], Bv[:s])
                if len(x) > 10 and np.std(x) > 0 and np.std(y2) > 0:
                    r = float(np.corrcoef(x, y2)[0, 1])
                    if abs(r) > abs(best):
                        best, lag = r, s
        return {"a": f"{a_topic}[{a_instance}].{a_field}", "b": f"{b_topic}[{b_instance}].{b_field}", "samples": int(len(grid)),
                "r_at_zero_lag": _r(r0, 3), "best_r": _r(best, 3), "best_lag_s": _r(lag * dt, 3),
                "note": "Correlation is not causation."}

    def get_physics(self) -> Dict[str, Any]:
        ph = self.a.get("physics") or {}
        dyn = dict(ph.get("dynamics") or {})
        if "windows" in dyn:
            dyn["windows"] = dyn["windows"][:30]
        return {"hover_balance": ph.get("hover_balance"), "thrust_residual": ph.get("thrust_residual"), "dynamics": dyn,
                "airframe": {k: (ph.get("airframe") or {}).get(k) for k in ("mass_kg", "arm_projection_m", "yaw_torque_coeff_cq",
                                                                             "motor_time_constant_s", "hover_throttle", "motors")}}

    def controller_tracking(self, controller: str, axis: str, t_start: Optional[float] = None,
                            t_end: Optional[float] = None) -> Dict[str, Any]:
        fl = self.src.fl  # type: ignore[attr-defined]
        cid, aid = CONTROLLER_IDS[controller], AXIS_IDS[axis]
        inst = next((i for i in fl.instances("arjuna_controller")
                     if len(fl.get("arjuna_controller", i)) and int(fl.get("arjuna_controller", i)["controller_id"].iloc[0]) == cid
                     and int(fl.get("arjuna_controller", i)["axis"].iloc[0]) == aid), None)
        if inst is None:
            return {"error": f"No {controller} controller record for axis {axis} in this log."}
        df = fl.get("arjuna_controller", inst)
        t = df["t"].to_numpy(float)
        m = self._window(t, t_start, t_end, None)
        scale = 180 / np.pi if cid in (0, 1) else 1.0
        unit = {0: "deg/s", 1: "deg"}.get(cid, "loop units")
        out: Dict[str, Any] = {"controller": controller, "axis": axis, "instance": inst, "unit": unit, "samples": int(m.sum())}
        if not m.any():
            return {**out, "note": "no samples in this window"}
        e = df["error"].to_numpy(float)[m] * scale
        e = e[np.isfinite(e)]
        if len(e):
            out.update(error_rms=_r(np.sqrt(np.mean(e ** 2))), error_p95_abs=_r(np.percentile(np.abs(e), 95)),
                       error_max_abs=_r(np.max(np.abs(e))))
        for k in ("setpoint", "measured"):
            v = df[k].to_numpy(float)[m] * scale
            if np.isfinite(v).any():
                out[f"{k}_range"] = [_r(np.nanmin(v)), _r(np.nanmax(v))]
        for k in ("p_term", "i_term", "d_term", "ff_term"):
            if k in df:
                v = df[k].to_numpy(float)[m]
                if np.isfinite(v).any():
                    out[f"{k}_rms"] = _r(np.sqrt(np.nanmean(v ** 2)))
        if "saturated" in df:
            out["fraction_saturated"] = _r(np.nanmean(df["saturated"].to_numpy(float)[m]), 4)
        if "integrator" in df and np.isfinite(df["integrator"].to_numpy(float)[m]).any():
            iv = df["integrator"].to_numpy(float)[m]
            iv = iv[np.isfinite(iv)]
            out["integrator_start_end"] = [_r(iv[0]), _r(iv[-1])]
        return out

    def motor_outputs(self, t_start: Optional[float] = None, t_end: Optional[float] = None) -> Dict[str, Any]:
        fl = self.src.fl  # type: ignore[attr-defined]
        ao = fl.get("arjuna_actuator_outputs")
        if ao is None:
            return {"error": "No arjuna_actuator_outputs in this log."}
        t = ao["t"].to_numpy(float)
        m = self._window(t, t_start, t_end, None)
        if not m.any():
            return {"note": "no samples in this window"}
        from src.arjuna.physics import motor_names
        names = motor_names(fl)
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.1
        pmax = float(fl.params.get("MOT_PWM_MAX", 2000) or 2000)
        motors, means = [], []
        for i, nm in enumerate(names):
            c = ao.get(f"motor_cmd[{i}]")
            if c is None:
                continue
            cv = c.to_numpy(float)[m]
            pw = ao[f"pwm_us[{i}]"].to_numpy(float)[m] if f"pwm_us[{i}]" in ao else np.full_like(cv, np.nan)
            means.append(np.nanmean(cv))
            motors.append({"motor": nm, "cmd_mean": _r(np.nanmean(cv)), "cmd_max": _r(np.nanmax(cv)), "pwm_mean": _r(np.nanmean(pw)),
                           "pwm_max": _r(np.nanmax(pw)), "time_at_max_pwm_s": _r(np.sum(pw >= pmax - 25) * dt, 2)})
        mean = float(np.mean(means)) if means else np.nan
        for mo, mv in zip(motors, means):
            mo["cmd_mean_vs_motor_average_pct"] = _r((mv / mean - 1) * 100, 2) if mean else None
        out = {"window": [t_start, t_end], "motors": motors, "pwm_max_param": pmax,
               "spread_max_minus_min_cmd_mean": _r(max(means) - min(means)) if means else None}
        if "thrust_sp" in ao:
            out["thrust_demand"] = self._stats(t[m], ao["thrust_sp"].to_numpy(float)[m], "0-1")
        if "mixer_saturated" in ao:
            out["mixer_saturated_s"] = _r(np.sum(np.nan_to_num(ao["mixer_saturated"].to_numpy(float)[m]) > 0) * dt, 2)
        return out

    def get_messages(self, t_start: Optional[float] = None, t_end: Optional[float] = None,
                     min_level: str = "info") -> Dict[str, Any]:
        fl = self.src.fl  # type: ignore[attr-defined]
        lo, hi = self.src.bounds()
        a, b = (lo if t_start is None else t_start), (hi if t_end is None else t_end)
        msgs = [{**mm, "at": clock(mm["t"])} for mm in fl.messages
                if a <= mm["t"] <= b and LEVEL_RANK.get(mm["level"], 1) >= LEVEL_RANK.get(min_level, 1)]
        events = [e for e in (self.a.get("timeline") or {}).get("events", []) if a <= (e.get("t") or -1) <= b]
        return {"window": [a, b], "messages": msgs[:80], "messages_total": len(msgs), "events": events[:80]}

    def _key_signals(self) -> List[Tuple[str, str, int, str]]:
        fl = self.src.fl  # type: ignore[attr-defined]
        out: List[Tuple[str, str, int, str]] = []

        def add(topic: str, field: str, inst: int = 0, label: str = "") -> None:
            df = fl.get(topic, inst)
            if df is not None and field in df and np.isfinite(df[field].to_numpy(float)).any():
                out.append((topic, field, inst, label or f"{topic.replace('arjuna_', '')}.{field}"))

        for i in range(3):
            add("arjuna_vibration", f"accel_vibe[{i}]", 0, f"vibration {'xyz'[i]}")
        add("arjuna_vibration", "clip_rate", 0, "IMU clipping rate")
        for f, lab in (("roll", "roll"), ("pitch", "pitch")):
            add("arjuna_attitude", f, 0, lab)
        for i, ax in enumerate(("roll", "pitch", "yaw")):
            add("arjuna_attitude", f"rates[{i}]", 0, f"{ax} rate")
        for inst in fl.instances("arjuna_controller"):
            df = fl.get("arjuna_controller", inst)
            if len(df) and int(df["controller_id"].iloc[0]) in (0, 1) and int(df["axis"].iloc[0]) in (0, 1, 2):
                loop = {0: "rate", 1: "attitude"}[int(df["controller_id"].iloc[0])]
                add("arjuna_controller", "error", inst, f"{loop} {['roll', 'pitch', 'yaw'][int(df['axis'].iloc[0])]} error")
        for i in range(4):
            add("arjuna_actuator_outputs", f"motor_cmd[{i}]", 0, f"motor {i + 1} command")
        add("arjuna_actuator_outputs", "thrust_sp", 0, "thrust demand")
        add("arjuna_battery", "voltage", 0, "battery voltage")
        add("arjuna_battery", "current", 0, "battery current")
        add("arjuna_position", "pos_u", 0, "height (estimator)")
        add("arjuna_position", "vel_u", 0, "climb rate (estimator)")
        add("arjuna_baro", "altitude", 0, "baro altitude")
        add("arjuna_gps", "h_acc", 0, "GPS h_acc")
        add("arjuna_gps", "satellites", 0, "GPS satellites")
        for inst in fl.instances("arjuna_estimator_innov"):
            add("arjuna_estimator_innov", "test_ratio", inst, f"estimator test ratio {inst}")
        for i in range(4):
            add("arjuna_rc_input", f"channels[{i}]", 0, self.src._rc_meaning(i))  # type: ignore[attr-defined]
        return out

    def sequence_of_departures(self, t_start: float, t_end: float, baseline_s: float = 60.0) -> Dict[str, Any]:
        b0, b1 = float(t_start) - float(baseline_s), float(t_start)
        rows, skipped = [], []
        for topic, field, inst, label in self._key_signals():
            t, y, unit = self.src.series(topic, field, inst)
            base = (t >= b0) & (t < b1) & np.isfinite(y)
            win = (t >= t_start) & (t <= t_end) & np.isfinite(y)
            if base.sum() < 10 or win.sum() < 2:
                skipped.append(label)
                continue
            med = float(np.median(y[base]))
            sig = float(np.median(np.abs(y[base] - med)) * 1.4826)
            sig = max(sig, 1e-3 * max(1.0, abs(med)))
            z = (y - med) / sig
            dt = float(np.median(np.diff(t[win]))) if win.sum() > 1 else 0.1
            need = max(2, int(np.ceil(DEPARTURE_MIN_S / max(dt, 1e-3))))
            hit = (np.abs(z) > DEPARTURE_Z) & win
            idx = np.where(hit)[0]
            onset = None
            for k in idx:
                if hit[k: k + need].all() or (k + need > len(hit) and hit[k:].all()):
                    onset = int(k)
                    break
            if onset is None:
                continue
            wi = np.where(win)[0]
            pk = int(wi[np.argmax(np.abs(z[wi]))])
            rows.append({"signal": label, "topic": topic, "field": field, "instance": inst, "unit": unit,
                         **({"approved_limits": _limits(topic, field)} if _limits(topic, field) else {}),
                         "onset_t": _r(t[onset], 2), "onset_at": clock(t[onset]), "baseline_median": _r(med),
                         "value_at_onset": _r(y[onset]), "peak_value": _r(y[pk]), "peak_t": _r(t[pk], 2),
                         "peak_robust_z": _r(z[pk], 1), "direction": "up" if z[pk] > 0 else "down"})
        rows.sort(key=lambda r: (r["onset_t"], -abs(r["peak_robust_z"] or 0)))
        return {"window": [_r(t_start, 2), _r(t_end, 2)], "baseline": [_r(b0, 2), _r(b1, 2)],
                "rule": f"onset = first time |value - baseline median| > {DEPARTURE_Z} robust sigmas for at least {DEPARTURE_MIN_S} s",
                "departures_in_time_order": rows, "no_departure": [r for r in [l for *_, l in self._key_signals()]
                                                                   if r not in {x["signal"] for x in rows} and r not in skipped],
                "not_enough_baseline": skipped,
                "note": "Earliest is not automatically the cause: a pilot input or mode change can come first. Check the "
                        "pilot sticks and modes in the same list."}

    def state_at(self, times: List[float]) -> Dict[str, Any]:
        fl = self.src.fl  # type: ignore[attr-defined]
        from src.arjuna.ulog import enum_name
        snaps = []
        for tq in [float(x) for x in (times or [])][:6]:
            snap: Dict[str, Any] = {"t": _r(tq, 2), "at": clock(tq)}
            vs = fl.get("arjuna_vehicle_status")
            if vs is not None and len(vs):
                i = int(np.clip(np.searchsorted(vs["t"].to_numpy(float), tq, side="right") - 1, 0, len(vs) - 1))
                snap["mode"] = enum_name("FLIGHT_MODE", vs["flight_mode"].iloc[i]) if "flight_mode" in vs else None
                snap["armed"] = bool(vs["armed"].iloc[i] > 0) if "armed" in vs and np.isfinite(vs["armed"].iloc[i]) else None
                snap["landed_state"] = enum_name("LANDED_STATE", vs["landed_state"].iloc[i]) if "landed_state" in vs else None
            for topic, field, inst, label in self._key_signals():
                if label.startswith(("vibration", "IMU", "estimator", "GPS", "rate ", "attitude ")):
                    continue
                t, y, unit = self.src.series(topic, field, inst)
                ok = np.isfinite(y)
                if not ok.any():
                    continue
                j = int(np.argmin(np.abs(t[ok] - tq)))
                if abs(t[ok][j] - tq) <= 2.0:
                    snap[label] = {"value": _r(y[ok][j]), "unit": unit}
            snaps.append(snap)
        return {"snapshots": snaps, "note": "Nearest sample within 2 s of each time; missing keys were not logged then."}

    def match_pilot_report(self) -> Dict[str, Any]:
        if not self.pilot:
            return {"error": "No pilot report was given for this flight."}
        tl = self.a.get("timeline") or {}
        lo, hi = self.src.bounds()
        return symptoms.rank(self.pilot, self.a.get("findings", []), tl.get("takeoff_t"), tl.get("landing_t"), hi)
