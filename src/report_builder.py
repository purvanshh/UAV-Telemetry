"""Build a validated FlightReport from Phase 2 analysis outputs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from src.event_builder import DEFAULT_MAX_GAP_S, AnomalyEvent, build_events
from src.event_metrics import enrich_event_dict
from src.report_schema import (
    SCHEMA_VERSION,
    AnomalyEventReport,
    DetectorInfo,
    FlightReport,
    FlightSummary,
    validate_report,
)


def build_flight_report(
    *,
    flight_id: str,
    analysis_status: str,
    windows: Optional[pd.DataFrame],
    threshold: float,
    variant: str,
    features: List[str],
    win_samples: int = 30,
    stride_samples: int = 15,
    feature_schema_version: Optional[str] = None,
    data_quality_warnings: Optional[List[str]] = None,
    errors: Optional[List[str]] = None,
    aligned: Optional[pd.DataFrame] = None,
    max_gap_s: float = DEFAULT_MAX_GAP_S,
    score_mean: Optional[float] = None,
    score_p50: Optional[float] = None,
    score_p95: Optional[float] = None,
    score_max: Optional[float] = None,
) -> FlightReport:
    """Assemble and validate the downloadable JSON report."""
    warnings = list(data_quality_warnings or [])
    err_list = list(errors or [])

    total = int(len(windows)) if windows is not None else 0
    flagged = (
        int(windows["is_anomaly"].astype(bool).sum())
        if windows is not None and total > 0
        else 0
    )
    flagged_rate = (flagged / total) if total > 0 else None

    events: List[AnomalyEvent] = []
    if windows is not None and total > 0 and flagged > 0:
        events = build_events(windows, max_gap_s=max_gap_s, features=features)

    # Status refinement
    status = analysis_status
    if status == "completed" and flagged == 0 and total > 0:
        status = "no_anomaly"

    event_payloads = [
        enrich_event_dict(
            ev,
            threshold=threshold,
            features=features,
            aligned=aligned,
            data_quality_warnings=[],  # keep flight-level DQ separate
        )
        for ev in events
    ]

    report = FlightReport(
        schema_version=SCHEMA_VERSION,
        flight_id=flight_id or "unknown",
        analysis_status=status,  # type: ignore[arg-type]
        detector=DetectorInfo(
            model="LSTMAE",
            variant=variant or "conservative",
            feature_count=len(features),
            window_length=win_samples,
            stride=stride_samples,
            threshold=float(threshold),
            features=list(features),
            feature_schema_version=feature_schema_version,
        ),
        summary=FlightSummary(
            total_windows=total,
            flagged_windows=flagged,
            flagged_rate=flagged_rate,
            event_count=len(events),
            score_mean=score_mean,
            score_p50=score_p50,
            score_p95=score_p95,
            score_max=score_max,
        ),
        events=[AnomalyEventReport.model_validate(p) for p in event_payloads],
        data_quality_warnings=warnings,
        grouping_policy={
            "rule": (
                "Merge flagged windows within the same session when intervals "
                "overlap or the gap between previous event end and next window "
                "start is <= max_gap_s. Never merge across session_id."
            ),
            "max_gap_s": max_gap_s,
            "duration_definition": "max(t_end) - min(t_start) of member windows",
        },
        ai_analysis=None,
        errors=err_list,
    )
    return validate_report(report)


def events_dataframe_from_report(report: FlightReport) -> pd.DataFrame:
    rows = []
    for ev in report.events:
        rows.append(
            {
                "event_id": ev.event_id,
                "session_id": ev.session_id,
                "start_time_s": ev.start_time_s,
                "end_time_s": ev.end_time_s,
                "duration_s": ev.duration_s,
                "flagged_window_count": ev.flagged_window_count,
                "peak_score": ev.peak_score,
                "mean_flagged_score": ev.mean_flagged_score,
                "threshold": ev.threshold,
                "peak_over_threshold": ev.peak_over_threshold,
                "threshold_relative_score": ev.threshold_relative_score,
                "affected_signals": ",".join(ev.affected_signals),
                "window_indices": ",".join(str(i) for i in ev.window_indices),
            }
        )
    return pd.DataFrame(rows)
