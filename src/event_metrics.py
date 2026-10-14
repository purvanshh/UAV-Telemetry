"""Derived metrics and evidence helpers for anomaly events."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.event_builder import AnomalyEvent


def threshold_relative_score(peak_score: float, threshold: float) -> Optional[float]:
    """Return peak / threshold when both are finite and threshold > 0."""
    if threshold is None or threshold <= 0:
        return None
    if peak_score is None or peak_score != peak_score:
        return None
    return float(peak_score / threshold)


def top_affected_signals(
    event: AnomalyEvent,
    *,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Rank signals by peak per-feature reconstruction error.

    Returns an empty list when no per-feature evidence exists — callers must
    not invent attributions.
    """
    if not event.per_feature_peak:
        return []
    ranked = sorted(
        event.per_feature_peak.items(), key=lambda kv: kv[1], reverse=True
    )
    out = []
    for name, peak in ranked[:top_k]:
        out.append(
            {
                "signal": name,
                "peak_recon_error": float(peak),
                "mean_recon_error": float(
                    event.per_feature_mean.get(name, peak)
                ),
            }
        )
    return out


def local_signal_summary(
    aligned: Optional[pd.DataFrame],
    event: AnomalyEvent,
    features: List[str],
    *,
    pad_s: float = 1.5,
) -> List[Dict[str, Any]]:
    """Summarize aligned telemetry in [start-pad, end+pad] for available features.

    Only includes features present in ``aligned``. Values are descriptive
    statistics, not causal claims.
    """
    if aligned is None or aligned.empty or "_t_s" not in aligned.columns:
        return []
    t0 = event.start_time_s - pad_s
    t1 = event.end_time_s + pad_s
    mask = (aligned["_t_s"] >= t0) & (aligned["_t_s"] <= t1)
    sub = aligned.loc[mask]
    if sub.empty:
        return []

    evidence = []
    for feat in features:
        if feat not in sub.columns:
            continue
        v = pd.to_numeric(sub[feat], errors="coerce").to_numpy(dtype="float64")
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        evidence.append(
            {
                "signal": feat,
                "n_samples": int(len(v)),
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
                "min": float(np.min(v)),
                "max": float(np.max(v)),
            }
        )
    return evidence


def enrich_event_dict(
    event: AnomalyEvent,
    *,
    threshold: float,
    features: List[str],
    aligned: Optional[pd.DataFrame] = None,
    data_quality_warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build the event payload used in the JSON report."""
    peak = event.peak_score
    affected = top_affected_signals(event)
    evidence = []
    # Prefer reconstruction-error ranking as primary evidence when available
    for item in affected:
        evidence.append(
            {
                "type": "per_feature_recon_error",
                "signal": item["signal"],
                "peak_recon_error": item["peak_recon_error"],
                "mean_recon_error": item["mean_recon_error"],
            }
        )
    for item in local_signal_summary(aligned, event, features):
        evidence.append({"type": "local_telemetry_summary", **item})

    return {
        "event_id": event.event_id,
        "session_id": event.session_id,
        "start_time_s": event.start_time_s,
        "end_time_s": event.end_time_s,
        "duration_s": event.duration_s,
        "peak_score": peak if peak == peak else None,
        "mean_flagged_score": (
            event.mean_flagged_score if event.scores else None
        ),
        "flagged_window_count": event.flagged_window_count,
        "threshold": threshold,
        "peak_over_threshold": (
            float(peak - threshold) if peak == peak else None
        ),
        "threshold_relative_score": threshold_relative_score(peak, threshold),
        "window_indices": list(event.window_indices),
        "affected_signals": [a["signal"] for a in affected],
        "evidence": evidence,
        "data_quality_warnings": list(data_quality_warnings or []),
    }
