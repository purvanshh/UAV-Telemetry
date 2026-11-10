"""Operator-facing views derived from verified analysis outputs.

Everything here is a *read-only projection* of what the backend already stored
(report, windows, aligned telemetry, data-quality accounting). Nothing in this module
re-scores, re-thresholds or modifies detector outputs, and no LLM is involved: all
wording is produced by fixed templates from measured values.

Two kinds of statements are kept apart and labelled as such:

* ``detector``  – the deployed LSTM autoencoder flagged these windows (authoritative).
* ``signal_checks`` – small rule-based checks on the stored telemetry (step change,
  sustained offset, flat-lined channel). They describe the *data*, use fixed
  heuristic limits, and have **not** been evaluated against labelled flights.

Patterns that are not implemented (oscillation, commanded-vs-measured mismatch) are
listed in :data:`NOT_IMPLEMENTED_CHECKS` and never reported.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.report_schema import FlightReport
from src.pilot_report import build_deep_pilot_report, deep_event_cards, friendly_signal_label

# ---- verdict -----------------------------------------------------------------------

ANOMALIES = "anomalies_detected"
NO_ALERTS_OK = "no_alerts_adequate_coverage"
NO_ALERTS_LOW = "no_alerts_low_coverage"
FAILED = "analysis_failed"
INSUFFICIENT = "insufficient_data"
NOT_ANALYZED = "not_analyzed"

VERDICT_STATES = (ANOMALIES, NO_ALERTS_OK, NO_ALERTS_LOW, FAILED, INSUFFICIENT, NOT_ANALYZED)

NOT_A_SAFETY_STATEMENT = (
    "This tool only compares recorded telemetry with patterns it was trained on. "
    "The absence of an alert does not show that the aircraft was safe or serviceable."
)

NOT_IMPLEMENTED_CHECKS = {
    "oscillation": "Repeated-oscillation check is not implemented.",
    "command_tracking": (
        "Commanded-versus-measured motion check is not implemented for operator "
        "findings (signal semantics are not verified for this data)."
    ),
}

SIGNAL_INFO: Dict[str, Dict[str, Optional[str]]] = {
    "gyro_x": {"label": "angular rate about the X axis (gyro X)", "unit": "rad/s"},
    "gyro_y": {"label": "angular rate about the Y axis (gyro Y)", "unit": "rad/s"},
    "gyro_z": {"label": "angular rate about the Z axis (gyro Z)", "unit": "rad/s"},
    "acc_x": {"label": "acceleration along X (accelerometer X)", "unit": "m/s²"},
    "acc_y": {"label": "acceleration along Y (accelerometer Y)", "unit": "m/s²"},
    "acc_z": {"label": "acceleration along Z (accelerometer Z)", "unit": "m/s²"},
    "roll": {"label": "measured roll angle", "unit": "deg"},
    "pitch": {"label": "measured pitch angle", "unit": "deg"},
    "yaw_sin": {"label": "heading (yaw, sine component)", "unit": None},
    "yaw_cos": {"label": "heading (yaw, cosine component)", "unit": None},
    "roll_res": {"label": "difference between measured and desired roll", "unit": "deg"},
    "pitch_res": {"label": "difference between measured and desired pitch", "unit": "deg"},
}

VERDICT_TEXT = {
    ANOMALIES: ("Unusual telemetry detected", "attention"),
    NO_ALERTS_OK: ("No alerts detected", "neutral"),
    NO_ALERTS_LOW: ("No alerts detected, but coverage was limited", "caution"),
    FAILED: ("Analysis failed", "error"),
    INSUFFICIENT: ("Not enough data to assess", "caution"),
    NOT_ANALYZED: ("Not analysed yet", "neutral"),
}


def compute_verdict(
    *,
    ok: Optional[bool],
    status: Optional[str],
    n_events: int,
    coverage_status: Optional[str],
    analysed_fraction: Optional[float] = None,
    errors: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Map verified analysis facts to one of the five operator states.

    "No flagged windows" is never reported as normal: with anything short of
    verified adequate coverage the state is ``no_alerts_low_coverage``.
    """
    if status is None:
        state = NOT_ANALYZED
    elif status == "insufficient_data" or coverage_status == "insufficient_data":
        state = INSUFFICIENT
    elif not ok or status == "error":
        state = FAILED
    elif n_events > 0:
        state = ANOMALIES
    elif coverage_status == "ok":
        state = NO_ALERTS_OK
    else:
        state = NO_ALERTS_LOW

    caveats: List[str] = []
    if state == ANOMALIES and coverage_status is None:
        caveats.append(
            "Data coverage was not recorded for this analysis, so how much of the flight "
            "was assessed cannot be confirmed. Re-run the analysis to record it."
        )
    elif state == ANOMALIES and coverage_status != "ok":
        caveats.append("Part of the flight could not be assessed (see coverage).")
    if state == NO_ALERTS_LOW:
        if coverage_status in (None, "unknown"):
            caveats.append(
                "Data coverage was not recorded for this analysis, so how much of the "
                "flight was assessed cannot be confirmed."
            )
        else:
            frac = f" Only {analysed_fraction:.0%} of the usable flight was assessed." if analysed_fraction is not None else ""
            caveats.append("Coverage was low, so this result says nothing about unassessed periods." + frac)
    if state in (NO_ALERTS_OK, NO_ALERTS_LOW):
        caveats.append(NOT_A_SAFETY_STATEMENT)

    headline, tone = VERDICT_TEXT[state]
    detail = {
        ANOMALIES: f"{n_events} period(s) of unusual telemetry were flagged for review.",
        NO_ALERTS_OK: "The detector flagged nothing in the part of the flight it could assess.",
        NO_ALERTS_LOW: "The detector flagged nothing in the part of the flight it could assess, "
        "but that part was limited or its extent is unknown.",
        FAILED: "The analysis did not complete. No conclusion can be drawn about this flight.",
        INSUFFICIENT: "The recording was too short, too fragmented, or missing signals the detector needs.",
        NOT_ANALYZED: "Run the analysis to see results.",
    }[state]
    return {
        "state": state,
        "headline": headline,
        "tone": tone,
        "detail": detail,
        "caveats": caveats,
        "errors": list(errors or []),
    }


def verdict_from_summary(summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Verdict from a stored ``analysis_summary.json`` (no report needed)."""
    if not summary:
        return compute_verdict(ok=None, status=None, n_events=0, coverage_status=None)
    dq = summary.get("data_quality") or {}
    return compute_verdict(
        ok=summary.get("ok"),
        status=summary.get("status"),
        n_events=int(summary.get("n_events") or 0),
        coverage_status=summary.get("coverage_status") or dq.get("coverage_status"),
        analysed_fraction=(dq.get("durations") or {}).get("analysed_fraction_of_aligned"),
        errors=summary.get("errors"),
    )


def verdict_from_report(report: FlightReport, summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    dq = report.data_quality or {}
    status = report.analysis_status
    ok = status in ("completed", "no_anomaly")
    cov = dq.get("coverage_status")
    # A stored summary may know more (e.g. coverage added later); never less.
    if summary:
        cov = cov or summary.get("coverage_status")
    # A report written before coverage accounting existed has no coverage -> unverified.
    return compute_verdict(
        ok=ok,
        status="insufficient_data" if status == "insufficient_data" else ("error" if status == "error" else "ok"),
        n_events=report.summary.event_count,
        coverage_status=cov,
        analysed_fraction=(dq.get("durations") or {}).get("analysed_fraction_of_aligned"),
        errors=report.errors,
    )


# ---- time helpers -------------------------------------------------------------------


def fmt_clock(rel_s: float) -> str:
    """``mm:ss`` (or ``h:mm:ss``) into the recording."""
    s = max(0, int(round(rel_s)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def _t0(report: FlightReport, aligned: Optional[pd.DataFrame]) -> float:
    tl = (report.data_quality or {}).get("timeline") or {}
    if tl.get("aligned_start_s") is not None:
        return float(tl["aligned_start_s"])
    if aligned is not None and len(aligned):
        return float(aligned["_t_s"].iloc[0])
    return 0.0


def _safe(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# ---- rule-based signal checks ----------------------------------------------------------

BASELINE_SPAN_S = 30.0
BASELINE_GUARD_S = 2.0
MIN_BASELINE_SAMPLES = 50
STEP_MAD_MULT = 8.0
OFFSET_SIGMA_MULT = 4.0
OFFSET_FRACTION = 0.7
FLATLINE_MIN_SAMPLES = 10

CHECKS_DISCLAIMER = (
    "Rule-based checks on the stored telemetry. They are separate from the anomaly "
    "detector, use fixed heuristic limits, and have not been evaluated against labelled "
    "flights. They describe the recorded signal, not a cause."
)


def _mad_sigma(x: np.ndarray) -> float:
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


def _channel_context(aligned: pd.DataFrame, start: float, end: float, session_id: int):
    t = aligned["_t_s"].to_numpy()
    sid = aligned["_session_id"].to_numpy()
    in_sess = sid == session_id
    ev = in_sess & (t >= start) & (t <= end)
    base = in_sess & (t >= start - BASELINE_SPAN_S) & (t < start - BASELINE_GUARD_S)
    return ev, base


def signal_checks(
    aligned: Optional[pd.DataFrame], signal: str, start: float, end: float, session_id: int
) -> List[Dict[str, Any]]:
    """Run the three implemented checks for one channel around one event."""
    if aligned is None or signal not in aligned.columns or "_session_id" not in aligned.columns:
        return []
    ev_m, base_m = _channel_context(aligned, start, end, session_id)
    y = aligned[signal].to_numpy(dtype=float)
    t = aligned["_t_s"].to_numpy()
    ev = y[ev_m]
    ev = ev[np.isfinite(ev)]
    base = y[base_m]
    base = base[np.isfinite(base)]
    out: List[Dict[str, Any]] = []
    info = SIGNAL_INFO.get(signal, {"label": signal, "unit": None})
    unit = f" {info['unit']}" if info.get("unit") else ""
    label = info["label"]
    if len(ev) < 3:
        return out

    has_base = len(base) >= MIN_BASELINE_SAMPLES

    # 1. step-like change: largest one-step jump vs the usual step size before the event
    if has_base:
        db = np.diff(base)
        scale = max(_mad_sigma(db), 1e-9)
        idx = np.where(ev_m)[0]
        yy = y[idx]
        dd = np.diff(yy)
        if len(dd) and np.isfinite(dd).any():
            k = int(np.nanargmax(np.abs(dd)))
            if abs(dd[k]) / scale > STEP_MAD_MULT and abs(dd[k]) > 1e-9:
                tt = float(t[idx[k + 1]])
                out.append(
                    {
                        "kind": "sudden_change",
                        "signal": signal,
                        "time_s": tt,
                        "text": (
                            f"The {label} changed by {dd[k]:+.3g}{unit} in a single 0.1 s step, "
                            f"about {abs(dd[k]) / scale:.0f}× the typical step size in the "
                            "preceding recording."
                        ),
                    }
                )

    # 2. sustained offset from recent behaviour
    if has_base:
        bmed, bsig = float(np.median(base)), max(_mad_sigma(base), 1e-9)
        dev = (ev - bmed) / bsig
        share = float(np.mean(np.abs(dev) > 3.0)) if len(dev) else 0.0
        same_side = abs(float(np.mean(np.sign(dev[np.abs(dev) > 3.0])))) > 0.9 if share else False
        if abs(float(np.median(ev)) - bmed) / bsig > OFFSET_SIGMA_MULT and share >= OFFSET_FRACTION and same_side:
            out.append(
                {
                    "kind": "sustained_deviation",
                    "signal": signal,
                    "time_s": start,
                    "text": (
                        f"The {label} stayed at about {np.median(ev):.3g}{unit} for most of the "
                        f"event, compared with about {bmed:.3g}{unit} in the preceding "
                        f"{BASELINE_SPAN_S:.0f} s."
                    ),
                }
            )

    # 3. flat-lined channel: identical consecutive values while it normally varies
    if has_base and np.mean(np.diff(base) == 0) < 0.2:
        run = best = 0
        for v in np.diff(ev) == 0:
            run = run + 1 if v else 0
            best = max(best, run)
        if best + 1 >= FLATLINE_MIN_SAMPLES:
            out.append(
                {
                    "kind": "flatline",
                    "signal": signal,
                    "time_s": start,
                    "text": (
                        f"The {label} held an identical value for at least {(best + 1) / 10:.1f} s "
                        "although it normally varies. This can indicate a stuck or "
                        "not-updating sensor, or a data-logging problem."
                    ),
                }
            )
    return out


# ---- event description -------------------------------------------------------------------


def _channel_evidence(
    aligned: Optional[pd.DataFrame], signal: str, start: float, end: float, session_id: int, rel0: float
) -> Dict[str, Any]:
    info = SIGNAL_INFO.get(signal, {"label": signal, "unit": None})
    item: Dict[str, Any] = {"signal": signal, "label": info["label"], "pilot_label": friendly_signal_label(signal, deep=True),
                            "technical_name": signal, "unit": info["unit"], "during": None, "before": None}
    if aligned is None or signal not in aligned.columns:
        return item
    ev_m, base_m = _channel_context(aligned, start, end, session_id)
    y = aligned[signal].to_numpy(dtype=float)
    ev, base = y[ev_m], y[base_m]
    ev, base = ev[np.isfinite(ev)], base[np.isfinite(base)]
    if len(ev):
        item["during"] = {"min": _safe(ev.min()), "max": _safe(ev.max()), "mean": _safe(ev.mean()), "n": int(len(ev))}
    if len(base) >= MIN_BASELINE_SAMPLES:
        item["before"] = {
            "p05": _safe(np.percentile(base, 5)),
            "median": _safe(np.median(base)),
            "p95": _safe(np.percentile(base, 95)),
            "span_s": BASELINE_SPAN_S,
        }
    return item


def _excluded_near(report: FlightReport, start: float, end: float, margin: float = 2.0) -> List[Dict[str, Any]]:
    return [
        iv
        for iv in (report.data_quality or {}).get("excluded_intervals", [])
        if iv["end_s"] >= start - margin and iv["start_s"] <= end + margin
    ]


def describe_event(
    report: FlightReport,
    event_id: str,
    aligned: Optional[pd.DataFrame],
    *,
    verdict: Optional[Dict[str, Any]] = None,
    top_k: int = 4,
) -> Dict[str, Any]:
    ev = next((e for e in report.events if e.event_id == event_id), None)
    if ev is None:
        raise KeyError(event_id)
    rel0 = _t0(report, aligned)
    s_rel, e_rel = ev.start_time_s - rel0, ev.end_time_s - rel0
    signals = [s for s in ev.affected_signals][:top_k]
    chans = [_channel_evidence(aligned, s, ev.start_time_s, ev.end_time_s, ev.session_id, rel0) for s in signals]
    checks: List[Dict[str, Any]] = []
    for s in signals:
        for c in signal_checks(aligned, s, ev.start_time_s, ev.end_time_s, ev.session_id):
            c["time_rel_s"] = c["time_s"] - rel0
            checks.append(c)

    labels = [SIGNAL_INFO.get(s, {"label": s})["label"] for s in signals]
    if labels:
        observed = (
            f"For {ev.duration_s:.1f} s the anomaly detector found the recorded telemetry unlike the "
            f"flights it was trained on, mainly in: {'; '.join(labels)}."
        )
    else:
        observed = (
            f"For {ev.duration_s:.1f} s the anomaly detector found the recorded telemetry unlike the "
            "flights it was trained on. Per-signal attribution is not available."
        )
    if checks:
        observed += " Signal checks on the stored data: " + " ".join(c["text"] for c in checks)

    uncertain = [
        "The detector flags unusual combinations of signals. It does not identify a cause, and "
        "unusual is not the same as unsafe: a deliberate manoeuvre or rough air can look unusual.",
        "Only 12 attitude and inertial signals are scored. Altitude, pressure, motor outputs and "
        "radio inputs are not part of the detector.",
    ]
    near = _excluded_near(report, ev.start_time_s, ev.end_time_s)
    if near:
        uncertain.append(
            f"{len(near)} period(s) without usable data lie within 2 s of this event, so what "
            "happened just before or after it may be missing."
        )
    if aligned is not None and all(c["before"] is None for c in chans) and chans:
        uncertain.append("Not enough earlier flight data in this segment to compare against.")
    cov = (report.data_quality or {}).get("coverage_status")
    if cov in (None,):
        uncertain.append("Data coverage was not recorded for this analysis.")
    elif cov != "ok":
        uncertain.append(f"Overall data coverage was '{cov.replace('_', ' ')}'.")

    action = (
        f"Review the log and any video or pilot notes around {fmt_clock(s_rel)}–{fmt_clock(e_rel)}. "
        "Check whether the movement matches what was commanded or intended. If the aircraft "
        "behaved unexpectedly at this time, follow your organisation's inspection procedure."
    )
    return {
        "event_id": ev.event_id,
        "session_id": ev.session_id,
        "start_s": ev.start_time_s,
        "end_s": ev.end_time_s,
        "start_rel_s": s_rel,
        "end_rel_s": e_rel,
        "start_label": fmt_clock(s_rel),
        "end_label": fmt_clock(e_rel),
        "duration_s": ev.duration_s,
        "flagged_window_count": ev.flagged_window_count,
        "affected_signals": list(ev.affected_signals),
        "what_observed": observed,
        "when": f"{fmt_clock(s_rel)} to {fmt_clock(e_rel)} into the recording ({ev.duration_s:.1f} s), flight segment {ev.session_id + 1}.",
        "channels": chans,
        "signal_checks": checks,
        "signal_checks_note": CHECKS_DISCLAIMER,
        "detector_note": f"Flagged by the anomaly detector in {ev.flagged_window_count} consecutive analysis window(s).",
        "uncertain": uncertain,
        "suggested_action": action,
        "nearby_unassessed": near,
    }


def event_context(
    report: FlightReport,
    event_id: str,
    aligned: Optional[pd.DataFrame],
    *,
    pad_s: Optional[float] = None,
    top_k: int = 4,
    max_points: int = 1200,
) -> Dict[str, Any]:
    """Telemetry for the event's top channels with context before and after (read-only).

    The window is at least 10 s on each side (or twice the event length). Samples stay
    inside the event's own flight segment so a source gap is never drawn as a continuous line.
    """
    ev = next((e for e in report.events if e.event_id == event_id), None)
    if ev is None:
        raise KeyError(event_id)
    rel0 = _t0(report, aligned)
    pad = pad_s if pad_s is not None else max(10.0, 2.0 * ev.duration_s)
    out: Dict[str, Any] = {
        "event_id": ev.event_id, "start_rel_s": ev.start_time_s - rel0, "end_rel_s": ev.end_time_s - rel0,
        "window_start_rel_s": ev.start_time_s - pad - rel0, "window_end_rel_s": ev.end_time_s + pad - rel0,
        "time": [], "channels": [], "segment_limited": False,
    }
    if aligned is None or "_session_id" not in aligned.columns:
        return out
    t = aligned["_t_s"].to_numpy()
    m = (aligned["_session_id"].to_numpy() == ev.session_id) & (t >= ev.start_time_s - pad) & (t <= ev.end_time_s + pad)
    if not m.any():
        return out
    sub = aligned.loc[m]
    step = max(1, int(np.ceil(len(sub) / max_points)))
    sub = sub.iloc[::step]
    out["time"] = [float(x) - rel0 for x in sub["_t_s"]]
    out["segment_limited"] = bool(sub["_t_s"].iloc[0] > ev.start_time_s - pad + 0.5 or sub["_t_s"].iloc[-1] < ev.end_time_s + pad - 0.5)
    for sig in ev.affected_signals[:top_k]:
        if sig not in sub.columns:
            continue
        info = SIGNAL_INFO.get(sig, {"label": sig, "unit": None})
        vals = [_safe(v) for v in sub[sig]]
        out["channels"].append({"signal": sig, "label": friendly_signal_label(sig, deep=True),
                                "technical_name": sig, "unit": info["unit"], "values": vals})
    return out


# ---- overview / timeline / report --------------------------------------------------------


def _unassessed(report: FlightReport, rel0: float) -> List[Dict[str, Any]]:
    out = []
    for iv in (report.data_quality or {}).get("excluded_intervals", []):
        reason = iv.get("reason", "unknown")
        text = {
            "non_finite_feature_values": "Some signals were missing or invalid",
            "source_gap": "Gap in the recording",
        }.get(reason, "No usable data")
        out.append(
            {
                "start_s": iv["start_s"],
                "end_s": iv["end_s"],
                "start_rel_s": iv["start_s"] - rel0,
                "end_rel_s": iv["end_s"] - rel0,
                "duration_s": iv["end_s"] - iv["start_s"],
                "reason": reason,
                "text": text,
            }
        )
    return out


def _input_summary(meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """How a combined CSV was read (stored mapping record); None for other upload types."""
    c = meta.get("combined")
    if not c:
        return None
    cols = [
        {"source": x["source_column"], "canonical": x.get("canonical"), "unit": x.get("unit"),
         "expected_unit": x.get("expected_unit"), "unit_status": x.get("unit_status"), "method": x.get("method"),
         "status": x.get("validation_status"), "converted": bool(x.get("conversion")), "ignored": bool(x.get("ignored")),
         "profile_role": x.get("profile_role"), "semantic_status": x.get("semantic_status")}
        for x in (c.get("mapping") or {}).get("columns", [])
    ]
    t = (c.get("mapping") or {}).get("time")
    if t:
        cols.insert(0, {"source": t["source_column"], "canonical": "time", "unit": t.get("unit"), "expected_unit": "us",
                        "unit_status": t.get("unit_status"), "method": t.get("method"), "status": t.get("validation_status"),
                        "converted": True, "ignored": False})
    return {"filename": c.get("filename"), "rows": c.get("rows"), "profile_id": c.get("profile_id"),
            "mapping_version": c.get("mapping_version"), "llm": c.get("llm"), "columns": cols}


def build_overview(
    *,
    meta: Dict[str, Any],
    report: Optional[FlightReport],
    summary: Optional[Dict[str, Any]],
    aligned: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    flight_id = meta.get("flight_id")
    base = {
        "flight_id": flight_id,
        "created_at": meta.get("created_at"),
        "analyzed_at": meta.get("analyzed_at"),
        "source": meta.get("source"),
        "filenames": meta.get("filenames") or {},
        "upload_warnings": meta.get("warnings") or [],
        "input": _input_summary(meta),
    }
    if report is None:
        v = verdict_from_summary(summary)
        base.update(
            verdict=v, events=[], unassessed=[], coverage=None,
            duration={"flight_s": None, "analysed_s": None},
            summary_text=v["detail"], warnings=list((summary or {}).get("errors") or []),
        )
        return base

    v = verdict_from_report(report, summary)
    dq = report.data_quality or {}
    dur = dq.get("durations") or {}
    rel0 = _t0(report, aligned)
    events = []
    for e in report.events:
        d = describe_event(report, e.event_id, aligned, verdict=v)
        events.append(
            {
                "event_id": e.event_id,
                "start_rel_s": d["start_rel_s"], "end_rel_s": d["end_rel_s"],
                "start_label": d["start_label"], "end_label": d["end_label"],
                "duration_s": d["duration_s"], "session_id": e.session_id,
                "headline": _event_headline(d),
                "channels": [c["label"] for c in d["channels"]],
                "has_signal_checks": bool(d["signal_checks"]),
            }
        )
    events.sort(key=lambda x: x["start_rel_s"])
    cov_frac = dur.get("analysed_fraction_of_aligned")
    coverage = {
        "status": dq.get("coverage_status"),
        "analysed_fraction": cov_frac,
        "known": bool(dq),
        "scored_windows": (dq.get("windows") or {}).get("scored"),
        "candidate_windows": (dq.get("windows") or {}).get("candidate"),
        "missing_signals": (dq.get("features") or {}).get("missing", []),
    }
    parts = [v["detail"]]
    if cov_frac is not None and dur.get("analysed_duration_s") is not None:
        parts.append(
            f"{dur['analysed_duration_s']:.0f} s of {dur.get('aligned_duration_s', 0):.0f} s of aligned data were assessed ({cov_frac:.0%})."
        )
    base.update(
        verdict=v,
        events=events,
        unassessed=_unassessed(report, rel0),
        coverage=coverage,
        duration={"flight_s": dur.get("flight_duration_s"), "analysed_s": dur.get("analysed_duration_s"),
                  "aligned_s": dur.get("aligned_duration_s")},
        summary_text=" ".join(parts),
        warnings=list(report.data_quality_warnings) + list(report.errors),
    )
    return base


def _event_headline(d: Dict[str, Any]) -> str:
    if d["signal_checks"]:
        kinds = {c["kind"] for c in d["signal_checks"]}
        names = {"sudden_change": "sudden change", "sustained_deviation": "sustained offset", "flatline": "flat-lined signal"}
        return "Unusual telemetry with " + ", ".join(sorted(names[k] for k in kinds))
    if d["channels"]:
        return f"Unusual telemetry in {d['channels'][0]['label']}"
    return "Unusual telemetry"


def build_timeline(
    report: FlightReport, aligned: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    rel0 = _t0(report, aligned)
    dq = report.data_quality or {}
    tl = dq.get("timeline") or {}
    end = tl.get("aligned_end_s") or (float(aligned["_t_s"].iloc[-1]) if aligned is not None and len(aligned) else None)
    items: List[Dict[str, Any]] = []
    for e in report.events:
        d = describe_event(report, e.event_id, aligned)
        items.append(
            {
                "type": "event", "id": e.event_id, "start_rel_s": d["start_rel_s"], "end_rel_s": d["end_rel_s"],
                "duration_s": d["duration_s"], "session_id": e.session_id, "title": _event_headline(d),
                "channels": [c["label"] for c in d["channels"]], "signal_checks": [c["kind"] for c in d["signal_checks"]],
            }
        )
    for i, iv in enumerate(_unassessed(report, rel0)):
        items.append(
            {
                "type": "unassessed", "id": f"gap-{i + 1:03d}", "start_rel_s": iv["start_rel_s"], "end_rel_s": iv["end_rel_s"],
                "duration_s": iv["duration_s"], "session_id": None, "title": iv["text"], "channels": [], "signal_checks": [],
            }
        )
    items.sort(key=lambda x: (x["start_rel_s"], x["type"]))
    return {
        "time_reference": "seconds / mm:ss from the start of the aligned recording (log clock, not wall-clock time)",
        "total_s": (end - rel0) if end is not None else None,
        "items": items,
    }


def build_operator_report(
    *,
    meta: Dict[str, Any],
    report: FlightReport,
    summary: Optional[Dict[str, Any]],
    aligned: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    ov = build_overview(meta=meta, report=report, summary=summary, aligned=aligned)
    ev_details = [describe_event(report, e["event_id"], aligned) for e in ov["events"]]
    prov = report.provenance or {}
    appendix = {
        "note": "Technical detail for engineers. Not needed to understand the findings above.",
        "detector": report.detector.model_dump(),
        "scores": report.summary.model_dump(),
        "signal_reference": [
            {
                "feature": s,
                "pilot_label": friendly_signal_label(s, deep=True),
                "technical_label": (SIGNAL_INFO.get(s) or {}).get("label", s),
                "unit": (SIGNAL_INFO.get(s) or {}).get("unit"),
            }
            for s in report.detector.features
        ],
        "events": [
            {
                "event_id": e.event_id, "peak_score": e.peak_score, "mean_flagged_score": e.mean_flagged_score,
                "threshold": e.threshold, "peak_over_threshold": e.peak_over_threshold,
                "window_indices": e.window_indices,
                "affected_signals": list(e.affected_signals),
                "flagged_window_count": e.flagged_window_count,
                "per_feature_evidence": [x for x in e.evidence if x.get("type") == "per_feature_recon_error"],
            }
            for e in report.events
        ],
        "data_quality": report.data_quality,
        "provenance": prov,
        "grouping_policy": report.grouping_policy,
        "not_implemented_checks": NOT_IMPLEMENTED_CHECKS,
        "report_schema_version": report.schema_version,
    }
    pilot = build_deep_pilot_report(
        report=report,
        details=ev_details,
        coverage=ov["coverage"],
        verdict=ov["verdict"],
        analysed_s=ov["duration"].get("analysed_s"),
        flight_s=ov["duration"].get("flight_s"),
    )
    return {
        "overview": ov,
        "events": ev_details,
        "pilot_report": pilot,
        "pilot_events": deep_event_cards(ev_details),
        "unassessed": ov["unassessed"],
        "methodology": (
            "Telemetry from the IMU, attitude and barometer logs is aligned to a common 10 Hz timeline, "
            "split into recording segments wherever data has a gap, and cut into 3-second windows. A trained "
            "model (an LSTM autoencoder) reconstructs each window; windows it reconstructs poorly relative to "
            "a fixed limit are flagged, and adjacent flags are grouped into events. The model was trained on "
            "past flights, its real-world accuracy has not been established with reviewed flight labels, and it "
            "cannot say why something looks unusual."
        ),
        "model_version": {
            "model": report.detector.model, "variant": report.detector.variant,
            "feature_schema_version": report.detector.feature_schema_version,
            "artifact_fingerprints": (prov.get("artifacts") if isinstance(prov, dict) else None),
        },
        "disclaimer": NOT_A_SAFETY_STATEMENT,
        "technical_appendix": appendix,
    }
