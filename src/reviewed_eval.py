"""Evaluation against human-reviewed intervals of real flights.

Label vocabulary (no fault or root-cause labels are accepted):

``expected``           reviewed, representative behaviour
``unusual_telemetry``  reviewed, unusual telemetry (the "anomaly" class)
``data_quality_issue`` reviewed, a logging/sensor-data problem
``uncertain``          reviewed but undecided, or not reviewed

Time that no interval covers is **unreviewed** — never assumed normal. A flight
with no ``expected`` interval contributes nothing to false-alarm rates.

Matching (event level; overlapping windows are never independent):
flagged windows are first grouped into events by ``build_events`` (same
``max_gap_s`` as the product). An event *overlaps* an interval when
``event.start <= interval.end`` and ``event.end >= interval.start``.

* An ``unusual_telemetry`` interval is **detected** if ≥ 1 event overlaps it, else
  **missed**. Latency = ``max(0, t_end of first flagged window overlapping it −
  interval.start)``.
* An event is a **false alarm** iff it overlaps an ``expected`` interval and
  overlaps no ``unusual_telemetry``, ``data_quality_issue`` or ``uncertain``
  interval.
* An event overlapping only unreviewed time is **unadjudicated** (counted, never
  scored as right or wrong). Events on ``data_quality_issue`` intervals are
  reported separately.
* False alarms per flight-hour = false-alarm events / hours of *union* of
  ``expected`` intervals (excluding time also inside non-expected intervals).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.event_builder import DEFAULT_MAX_GAP_S, build_events
from src.threshold_sensitivity import score_distribution

SCHEMA = "reviewed_intervals/1"
LABELS = ("expected", "unusual_telemetry", "data_quality_issue", "uncertain")


class LabelError(ValueError):
    pass


def load_intervals(path: Path | str) -> Dict[str, Any]:
    m = json.loads(Path(path).read_text())
    return validate_intervals(m)


def validate_intervals(m: Mapping[str, Any]) -> Dict[str, Any]:
    if m.get("schema") != SCHEMA:
        raise LabelError(f"unsupported schema {m.get('schema')!r}; expected {SCHEMA}")
    ivs = list(m.get("intervals", []))
    if ivs and not (m.get("reviewer") and m.get("reviewed_at")):
        raise LabelError("labelled intervals require 'reviewer' and 'reviewed_at' (provenance of the labels)")
    by_flight: Dict[str, List[Mapping[str, Any]]] = {}
    for i, iv in enumerate(ivs):
        for k in ("flight_id", "t_start_s", "t_end_s", "label"):
            if k not in iv:
                raise LabelError(f"interval {i}: missing '{k}'")
        if iv["label"] not in LABELS:
            raise LabelError(f"interval {i}: label {iv['label']!r} not in {LABELS}")
        if not iv["t_end_s"] > iv["t_start_s"]:
            raise LabelError(f"interval {i}: t_end_s must exceed t_start_s")
        by_flight.setdefault(iv["flight_id"], []).append(iv)
    for fid, lst in by_flight.items():
        lst = sorted(lst, key=lambda x: x["t_start_s"])
        for a, b in zip(lst, lst[1:]):
            if b["t_start_s"] < a["t_end_s"] and a["label"] != b["label"]:
                raise LabelError(
                    f"{fid}: overlapping intervals with conflicting labels "
                    f"({a['label']} vs {b['label']})"
                )
    return dict(m)


def _union(ivs: Sequence[Sequence[float]]) -> List[List[float]]:
    out: List[List[float]] = []
    for s, e in sorted(ivs):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([float(s), float(e)])
    return out


def _subtract(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    res = []
    for s, e in a:
        cur = s
        for bs, be in b:
            if be <= cur or bs >= e:
                continue
            if bs > cur:
                res.append([cur, bs])
            cur = max(cur, be)
        if cur < e:
            res.append([cur, e])
    return res


def _ov(s0: float, e0: float, s1: float, e1: float) -> bool:
    return s0 <= e1 and e0 >= s1


def evaluate_flight(
    windows: pd.DataFrame,
    intervals: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
    max_gap_s: float = DEFAULT_MAX_GAP_S,
) -> Dict[str, Any]:
    w = windows.copy()
    w["is_anomaly"] = pd.to_numeric(w["anomaly_score"], errors="coerce") > float(threshold)
    events = build_events(w, max_gap_s=max_gap_s)
    by = {lab: [iv for iv in intervals if iv["label"] == lab] for lab in LABELS}
    unusual = by["unusual_telemetry"]
    detected, missed, latencies = [], [], []
    for iv in unusual:
        s, e = float(iv["t_start_s"]), float(iv["t_end_s"])
        hit = [ev for ev in events if _ov(ev.start_time_s, ev.end_time_s, s, e)]
        if hit:
            fw = w[w["is_anomaly"] & (w["t_start"] <= e) & (w["t_end"] >= s)]
            lat = max(0.0, float(fw["t_end"].min()) - s) if len(fw) else None
            detected.append({**dict(iv), "events": [ev.event_id for ev in hit], "latency_s": lat})
            if lat is not None:
                latencies.append(lat)
        else:
            missed.append(dict(iv))
    fa, unadj, dq_ev, unusual_ev, unc_ev = [], [], [], [], []
    for ev in events:
        def hits(lab: str) -> bool:
            return any(_ov(ev.start_time_s, ev.end_time_s, iv["t_start_s"], iv["t_end_s"]) for iv in by[lab])

        h_exp, h_unu, h_dq, h_unc = hits("expected"), hits("unusual_telemetry"), hits("data_quality_issue"), hits("uncertain")
        if h_unu:
            unusual_ev.append(ev.event_id)
        elif h_dq:
            dq_ev.append(ev.event_id)
        elif h_unc:
            unc_ev.append(ev.event_id)
        elif h_exp:
            fa.append(ev.event_id)
        else:
            unadj.append(ev.event_id)
    exp_u = _union([[iv["t_start_s"], iv["t_end_s"]] for iv in by["expected"]])
    other = _union([[iv["t_start_s"], iv["t_end_s"]] for lab in LABELS if lab != "expected" for iv in by[lab]])
    exp_clean = _subtract(exp_u, other)
    exp_hours = sum(e - s for s, e in exp_clean) / 3600.0
    return {
        "n_windows": int(len(w)),
        "n_flagged_windows": int(w["is_anomaly"].sum()),
        "n_events": len(events),
        "unusual_intervals": len(unusual),
        "detected": detected,
        "missed": missed,
        "latencies_s": latencies,
        "events_on_unusual": unusual_ev,
        "events_on_data_quality": dq_ev,
        "events_on_uncertain": unc_ev,
        "false_alarm_events": fa,
        "unadjudicated_events": unadj,
        "expected_hours": exp_hours,
    }


def evaluate(
    windows_by_flight: Mapping[str, pd.DataFrame],
    labels: Mapping[str, Any],
    *,
    threshold: float,
    max_gap_s: float = DEFAULT_MAX_GAP_S,
) -> Dict[str, Any]:
    """Aggregate over labelled flights. Returns ``status='no_reviewed_labels'`` when none."""
    ivs = labels.get("intervals", [])
    flights = sorted({iv["flight_id"] for iv in ivs} & set(windows_by_flight))
    unscored = sorted({iv["flight_id"] for iv in ivs} - set(windows_by_flight))
    if not flights:
        return {
            "status": "no_reviewed_labels",
            "reason": "No reviewed intervals match a scored flight. Real-world precision, recall, "
            "specificity and false-alarm rate cannot be computed and are not reported.",
            "labelled_flights_without_scores": unscored,
        }
    per = {
        f: evaluate_flight(windows_by_flight[f], [iv for iv in ivs if iv["flight_id"] == f],
                           threshold=threshold, max_gap_s=max_gap_s)
        for f in flights
    }
    n_unusual = sum(p["unusual_intervals"] for p in per.values())
    n_det = sum(len(p["detected"]) for p in per.values())
    exp_h = sum(p["expected_hours"] for p in per.values())
    n_fa = sum(len(p["false_alarm_events"]) for p in per.values())
    lat = [x for p in per.values() for x in p["latencies_s"]]
    return {
        "status": "ok",
        "threshold": float(threshold),
        "n_flights": len(flights),
        "n_windows": sum(p["n_windows"] for p in per.values()),
        "n_flagged_windows": sum(p["n_flagged_windows"] for p in per.values()),
        "reviewed_unusual_intervals": n_unusual,
        "event_recall": (n_det / n_unusual) if n_unusual else None,
        "missed_intervals": [m for p in per.values() for m in p["missed"]],
        "detection_latency_s": {
            "median": float(np.median(lat)) if lat else None,
            "max": float(np.max(lat)) if lat else None,
            "n": len(lat),
        },
        "expected_hours_reviewed": exp_h,
        "false_alarm_events": n_fa,
        "false_alarms_per_flight_hour": (n_fa / exp_h) if exp_h > 0 else None,
        "events_on_data_quality_intervals": sum(len(p["events_on_data_quality"]) for p in per.values()),
        "unadjudicated_events": sum(len(p["unadjudicated_events"]) for p in per.values()),
        "labelled_flights_without_scores": unscored,
        "per_flight": per,
    }


def per_feature_error_summary(windows: pd.DataFrame, threshold: float) -> Dict[str, Any]:
    """Mean per-feature reconstruction error for flagged vs unflagged windows (no labels needed)."""
    cols = [c for c in windows.columns if c.startswith("feat_err_")]
    flagged = windows["anomaly_score"] > float(threshold)
    out: Dict[str, Any] = {"n_flagged": int(flagged.sum()), "n_unflagged": int((~flagged).sum()), "features": {}}
    for c in cols:
        out["features"][c[len("feat_err_"):]] = {
            "mean_flagged": float(windows.loc[flagged, c].mean()) if flagged.any() else None,
            "mean_unflagged": float(windows.loc[~flagged, c].mean()) if (~flagged).any() else None,
        }
    return out


def descriptive(windows: pd.DataFrame, threshold: float) -> Dict[str, Any]:
    s = windows["anomaly_score"]
    return {
        "n_windows": int(len(s)),
        "flagged": int((s > threshold).sum()),
        "flagged_fraction": float((s > threshold).mean()) if len(s) else None,
        "score_distribution": score_distribution(s),
        "caveat": "Flagged fraction is NOT a false-alarm rate unless the flagged time is reviewed as expected.",
    }
