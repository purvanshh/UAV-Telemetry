"""Build a compact, event-centered evidence pack for the LLM.

Does not invent measurements. Detector scores/flags are copied as facts.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.report_schema import FlightReport

# Soft budget for serialized evidence JSON characters
DEFAULT_MAX_CHARS = 24_000
DEFAULT_MAX_EVENTS = 8
DEFAULT_BASELINE_WINDOWS = 5
DEFAULT_TOP_WINDOWS = 8


DETECTOR_LIMITATIONS = [
    "Deployed detector is a conservative 12-feature LSTM Autoencoder "
    "(IMU gyro/acc + ATT orientation residuals). It does not score BARO "
    "altitude, pressure, temperature, or rate features.",
    "Anomaly scores are reconstruction MSE vs a validation-calibrated "
    "threshold; they are not calibrated fault probabilities.",
    "Affected signals ranked by per-feature reconstruction error are "
    "attribution heuristics, not proven root causes.",
    "Data-quality warnings (e.g. extreme BARO) must not be treated as "
    "LSTM detections of pressure/altitude faults.",
    "When flagged_windows=0, high-scoring windows below threshold are "
    "context only — they are not confirmed anomalies and must not be "
    "labeled as detector events.",
]


def _round_floats(obj: Any, ndigits: int = 4) -> Any:
    if isinstance(obj, float):
        if obj != obj:  # NaN
            return None
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, ndigits) for v in obj]
    return obj


def _baseline_window_summaries(
    windows: Optional[pd.DataFrame],
    event_start: float,
    event_end: float,
    *,
    n: int = DEFAULT_BASELINE_WINDOWS,
) -> List[Dict[str, Any]]:
    """Nearby non-flagged windows before/after the event for comparison."""
    if windows is None or len(windows) == 0:
        return []
    if "is_anomaly" not in windows.columns:
        return []
    normal = windows[~windows["is_anomaly"].astype(bool)].copy()
    if normal.empty:
        return []
    # Prefer windows ending just before event, else after
    before = normal[normal["t_end"] <= event_start].tail(n)
    after = normal[normal["t_start"] >= event_end].head(max(0, n - len(before)))
    picked = pd.concat([before, after], ignore_index=True)
    out = []
    for _, row in picked.iterrows():
        out.append(
            {
                "window_index": int(row["window_index"]),
                "t_start": float(row["t_start"]),
                "t_end": float(row["t_end"]),
                "anomaly_score": float(row["anomaly_score"]),
                "is_anomaly": False,
            }
        )
    return out


def _aligned_slice_summary(
    aligned: Optional[pd.DataFrame],
    features: Sequence[str],
    t0: float,
    t1: float,
) -> List[Dict[str, Any]]:
    if aligned is None or aligned.empty or "_t_s" not in aligned.columns:
        return []
    sub = aligned[(aligned["_t_s"] >= t0) & (aligned["_t_s"] <= t1)]
    if sub.empty:
        return []
    stats = []
    for feat in features:
        if feat not in sub.columns:
            continue
        v = pd.to_numeric(sub[feat], errors="coerce").to_numpy(dtype="float64")
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        stats.append(
            {
                "signal": feat,
                "n": int(len(v)),
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
                "min": float(np.min(v)),
                "max": float(np.max(v)),
            }
        )
    return stats


def select_events_for_context(
    report: FlightReport,
    *,
    selected_event_ids: Optional[Sequence[str]] = None,
    max_events: int = DEFAULT_MAX_EVENTS,
) -> List[Any]:
    events = list(report.events)
    if selected_event_ids:
        wanted = set(selected_event_ids)
        events = [e for e in events if e.event_id in wanted]
    else:
        events = sorted(
            events,
            key=lambda e: (e.peak_score is not None, e.peak_score or 0.0),
            reverse=True,
        )
    return events[:max_events]


def _feat_err_columns(windows: pd.DataFrame, features: Sequence[str]) -> List[str]:
    cols = []
    for feat in features:
        c = f"feat_err_{feat}"
        if c in windows.columns:
            cols.append(c)
    return cols


def _top_windows_by_score(
    windows: Optional[pd.DataFrame],
    features: Sequence[str],
    *,
    threshold: float,
    n: int = DEFAULT_TOP_WINDOWS,
) -> List[Dict[str, Any]]:
    """Highest-scoring windows for context. Never re-label below-threshold as anomalies."""
    if windows is None or len(windows) == 0 or "anomaly_score" not in windows.columns:
        return []
    df = windows.copy()
    df["anomaly_score"] = pd.to_numeric(df["anomaly_score"], errors="coerce")
    df = df[np.isfinite(df["anomaly_score"].to_numpy(dtype="float64"))]
    if df.empty:
        return []
    df = df.sort_values("anomaly_score", ascending=False).head(n)
    feat_cols = _feat_err_columns(df, features)
    out: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        score = float(row["anomaly_score"])
        is_anom = bool(row["is_anomaly"]) if "is_anomaly" in df.columns else score > threshold
        top_feats: List[Dict[str, Any]] = []
        if feat_cols:
            errs = [(c.replace("feat_err_", ""), float(row[c])) for c in feat_cols]
            errs.sort(key=lambda x: x[1], reverse=True)
            for name, err in errs[:3]:
                top_feats.append({"signal": name, "recon_error": err})
        out.append(
            {
                "window_index": int(row["window_index"])
                if "window_index" in df.columns and pd.notna(row.get("window_index"))
                else None,
                "session_id": int(row["session_id"])
                if "session_id" in df.columns and pd.notna(row.get("session_id"))
                else None,
                "t_start": float(row["t_start"]) if "t_start" in df.columns else None,
                "t_end": float(row["t_end"]) if "t_end" in df.columns else None,
                "anomaly_score": score,
                "threshold": float(threshold),
                "is_anomaly": bool(is_anom),
                "below_threshold": bool(score <= threshold),
                "label": (
                    "flagged_anomaly_window"
                    if is_anom
                    else "unflagged_high_score_context_only"
                ),
                "top_feature_recon_errors": top_feats,
            }
        )
    return out


def _score_context(
    report: FlightReport, windows: Optional[pd.DataFrame]
) -> Dict[str, Any]:
    from src.diagnostic_trace import score_distribution

    dist = score_distribution(windows)
    thr = float(report.detector.threshold)
    score_max = dist.get("max")
    return {
        "threshold": thr,
        "flagged_windows": int(report.summary.flagged_windows),
        "event_count": int(report.summary.event_count),
        "score_distribution": dist,
        "margin_to_threshold": None
        if score_max is None
        else float(thr - float(score_max)),
        "no_threshold_exceedances": int(report.summary.flagged_windows) == 0,
        "note": (
            "Scores are reconstruction MSE on the 12 deployed IMU/ATT features. "
            "Windows with score <= threshold are not detector anomalies."
        ),
    }


def build_evidence_pack(
    report: FlightReport,
    *,
    windows: Optional[pd.DataFrame] = None,
    aligned: Optional[pd.DataFrame] = None,
    selected_event_ids: Optional[Sequence[str]] = None,
    max_events: int = DEFAULT_MAX_EVENTS,
    max_top_windows: int = DEFAULT_TOP_WINDOWS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> Dict[str, Any]:
    """Return a JSON-serializable evidence dict for prompts."""
    features = list(report.detector.features)
    selected = select_events_for_context(
        report, selected_event_ids=selected_event_ids, max_events=max_events
    )

    event_blocks = []
    for ev in selected:
        block = {
            "event_id": ev.event_id,
            "session_id": ev.session_id,
            "start_time_s": ev.start_time_s,
            "end_time_s": ev.end_time_s,
            "duration_s": ev.duration_s,
            "peak_score": ev.peak_score,
            "mean_flagged_score": ev.mean_flagged_score,
            "threshold": ev.threshold,
            "peak_over_threshold": ev.peak_over_threshold,
            "flagged_window_count": ev.flagged_window_count,
            "affected_signals": list(ev.affected_signals),
            "evidence": list(ev.evidence)[:12],
            "baseline_windows": _baseline_window_summaries(
                windows, ev.start_time_s, ev.end_time_s
            ),
            "imu_att_summary": _aligned_slice_summary(
                aligned,
                features,
                ev.start_time_s - 1.5,
                ev.end_time_s + 1.5,
            ),
        }
        event_blocks.append(block)

    thr = float(report.detector.threshold)
    top_windows = _top_windows_by_score(
        windows, features, threshold=thr, n=max_top_windows
    )
    score_ctx = _score_context(report, windows)

    instructions = [
        "Distinguish observed_facts (from this evidence) from hypotheses.",
        "Do not invent sensor values, timestamps, or detector outputs.",
        "Do not claim BARO/pressure/altitude caused the LSTM score.",
        "Use confidence only as low|medium|high — not as a probability.",
        "Never manufacture events or treat below-threshold windows as anomalies.",
    ]
    if not event_blocks:
        instructions.extend(
            [
                "No anomaly events were grouped. Explain why (flagged_windows=0, "
                "score_max vs threshold) using score_context and top_windows_by_score.",
                "Discuss data_quality_warnings as quality context, not as LSTM detections.",
                "List what additional evidence would be needed for a fault hypothesis.",
                "You may use findings with event_id 'flight_overview' for flight-level notes.",
                "Set insufficient_evidence=true for a specific physical fault when "
                "no threshold exceedances exist; still provide an explanatory summary.",
            ]
        )
    else:
        instructions.append(
            "If event evidence is thin, set insufficient_evidence=true."
        )

    pack = {
        "flight_id": report.flight_id,
        "analysis_status": report.analysis_status,
        "detector": {
            "model": report.detector.model,
            "variant": report.detector.variant,
            "feature_count": report.detector.feature_count,
            "features": features,
            "window_length": report.detector.window_length,
            "stride": report.detector.stride,
            "threshold": report.detector.threshold,
            "feature_schema_version": report.detector.feature_schema_version,
        },
        "summary": report.summary.model_dump(mode="json"),
        "score_context": score_ctx,
        "data_quality_warnings": list(report.data_quality_warnings),
        "detector_limitations": list(DETECTOR_LIMITATIONS),
        "events": event_blocks,
        "events_omitted": max(0, len(report.events) - len(event_blocks)),
        "top_windows_by_score": top_windows,
        "instructions_for_model": instructions,
    }
    pack = _round_floats(pack)

    # Truncate until within budget (events first, then top windows, then metadata)
    serialized = json.dumps(pack, ensure_ascii=False)
    guard = 0
    while len(serialized) > max_chars and guard < 40:
        guard += 1
        if pack["events"]:
            last = pack["events"][-1]
            if last.get("evidence"):
                last["evidence"] = last["evidence"][: max(0, len(last["evidence"]) // 2)]
            elif last.get("imu_att_summary"):
                last["imu_att_summary"] = last["imu_att_summary"][
                    : max(0, len(last["imu_att_summary"]) // 2)
                ]
            elif last.get("baseline_windows"):
                last["baseline_windows"] = []
            else:
                pack["events"].pop()
                pack["events_omitted"] = pack.get("events_omitted", 0) + 1
        elif pack.get("top_windows_by_score"):
            pack["top_windows_by_score"] = pack["top_windows_by_score"][
                : max(0, len(pack["top_windows_by_score"]) // 2)
            ]
            if not pack["top_windows_by_score"]:
                pack["context_truncated"] = True
        elif len(pack.get("detector_limitations") or []) > 1:
            pack["detector_limitations"] = pack["detector_limitations"][:1]
            pack["context_truncated"] = True
        elif len(pack.get("instructions_for_model") or []) > 1:
            pack["instructions_for_model"] = pack["instructions_for_model"][:1]
            pack["context_truncated"] = True
        else:
            # Last resort: keep a minimal stub under budget (retain score_context)
            feats = list(pack["detector"].get("features") or [])
            pack = {
                "flight_id": pack["flight_id"],
                "analysis_status": pack["analysis_status"],
                "detector": {
                    "variant": pack["detector"].get("variant"),
                    "feature_count": pack["detector"].get("feature_count"),
                    "features": feats,
                    "threshold": pack["detector"].get("threshold"),
                },
                "summary": {
                    "total_windows": pack["summary"].get("total_windows"),
                    "flagged_windows": pack["summary"].get("flagged_windows"),
                    "event_count": pack["summary"].get("event_count"),
                    "score_max": pack["summary"].get("score_max"),
                },
                "score_context": pack.get("score_context")
                or {
                    "threshold": pack["detector"].get("threshold"),
                    "no_threshold_exceedances": True,
                },
                "data_quality_warnings": list(pack.get("data_quality_warnings") or [])[
                    :3
                ],
                "detector_limitations": [
                    "Conservative 12-feature LSTM; BARO excluded from detection."
                ],
                "events": [],
                "events_omitted": pack.get("events_omitted", 0),
                "top_windows_by_score": [],
                "instructions_for_model": [
                    "Explain no-flag outcome from score_context; "
                    "set insufficient_evidence=true for a specific physical fault."
                ],
                "context_truncated": True,
            }
            serialized = json.dumps(pack, ensure_ascii=False)
            if len(serialized) > max_chars:
                pack["detector"]["features"] = []
                pack["detector_limitations"] = []
                pack["instructions_for_model"] = []
                pack["data_quality_warnings"] = []
                serialized = json.dumps(pack, ensure_ascii=False)
            if len(serialized) > max_chars:
                pack = {
                    "flight_id": pack["flight_id"],
                    "analysis_status": pack["analysis_status"],
                    "score_context": pack.get("score_context"),
                    "context_truncated": True,
                    "events": [],
                    "top_windows_by_score": [],
                }
                serialized = json.dumps(pack, ensure_ascii=False)
            break
        serialized = json.dumps(pack, ensure_ascii=False)

    pack["_approx_chars"] = len(serialized)
    return pack


def evidence_to_prompt_json(pack: Dict[str, Any]) -> str:
    clean = {k: v for k, v in pack.items() if not str(k).startswith("_")}
    return json.dumps(clean, indent=2, ensure_ascii=False)
