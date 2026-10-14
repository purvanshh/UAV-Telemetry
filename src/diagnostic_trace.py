"""Compact, non-sensitive diagnostic traces for UI / logs.

Never includes API keys or raw telemetry samples — only counts, score
distribution summaries, feature names, and evidence-pack metadata.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.report_schema import FlightReport


def _score_array(windows: Optional[pd.DataFrame]) -> np.ndarray:
    if windows is None or len(windows) == 0 or "anomaly_score" not in windows.columns:
        return np.array([], dtype=np.float64)
    return pd.to_numeric(windows["anomaly_score"], errors="coerce").to_numpy(
        dtype="float64"
    )


def score_distribution(windows: Optional[pd.DataFrame]) -> Dict[str, Any]:
    scores = _score_array(windows)
    finite = scores[np.isfinite(scores)]
    if len(finite) == 0:
        return {
            "n": 0,
            "min": None,
            "max": None,
            "mean": None,
            "p50": None,
            "p95": None,
        }
    return {
        "n": int(len(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "p50": float(np.quantile(finite, 0.50)),
        "p95": float(np.quantile(finite, 0.95)),
    }


def build_diagnostic_trace(
    report: FlightReport,
    *,
    windows: Optional[pd.DataFrame] = None,
    evidence_pack: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compact pipeline trace for display (safe to show / log)."""
    dist = score_distribution(windows)
    thr = float(report.detector.threshold)
    score_max = dist.get("max")
    margin = None if score_max is None else float(thr - score_max)

    n_events_in_pack = 0
    n_top_unflagged = 0
    approx_chars = None
    if evidence_pack is not None:
        n_events_in_pack = len(evidence_pack.get("events") or [])
        n_top_unflagged = len(evidence_pack.get("top_windows_by_score") or [])
        approx_chars = evidence_pack.get("_approx_chars")

    return {
        "flight_id": report.flight_id,
        "analysis_status": report.analysis_status,
        "scored_windows": int(report.summary.total_windows),
        "flagged_windows": int(report.summary.flagged_windows),
        "event_count": int(report.summary.event_count),
        "threshold": thr,
        "score_distribution": dist,
        "margin_to_threshold": margin,
        "features": list(report.detector.features),
        "feature_count": int(report.detector.feature_count),
        "variant": report.detector.variant,
        "window_length": report.detector.window_length,
        "stride": report.detector.stride,
        "data_quality_warning_count": len(report.data_quality_warnings or []),
        "evidence_events_sent": n_events_in_pack,
        "evidence_top_windows_sent": n_top_unflagged,
        "evidence_approx_chars": approx_chars,
        "notes": [
            "Flags use scores > saved threshold from threshold_config.json.",
            "top_windows_by_score below threshold are not confirmed anomalies.",
            "BARO signals are excluded from LSTM scoring.",
        ],
    }


def format_diagnostic_trace(trace: Dict[str, Any]) -> str:
    dist = trace.get("score_distribution") or {}
    lines: List[str] = [
        f"flight={trace.get('flight_id')} status={trace.get('analysis_status')}",
        (
            f"windows={trace.get('scored_windows')} flagged={trace.get('flagged_windows')} "
            f"events={trace.get('event_count')} threshold={trace.get('threshold')}"
        ),
        (
            f"scores min={dist.get('min')} p50={dist.get('p50')} "
            f"p95={dist.get('p95')} max={dist.get('max')} "
            f"margin_to_thr={trace.get('margin_to_threshold')}"
        ),
        (
            f"features={trace.get('feature_count')} variant={trace.get('variant')} "
            f"win={trace.get('window_length')} stride={trace.get('stride')}"
        ),
        (
            f"evidence events={trace.get('evidence_events_sent')} "
            f"top_windows={trace.get('evidence_top_windows_sent')} "
            f"chars≈{trace.get('evidence_approx_chars')} "
            f"dq_warnings={trace.get('data_quality_warning_count')}"
        ),
    ]
    return "\n".join(lines)
