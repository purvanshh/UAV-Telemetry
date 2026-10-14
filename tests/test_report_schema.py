"""Pydantic report schema validation."""

from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from src.report_builder import build_flight_report
from src.report_schema import FlightReport, validate_report


FEATURES = [
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "acc_x",
    "acc_y",
    "acc_z",
    "roll",
    "pitch",
    "yaw_sin",
    "yaw_cos",
    "roll_res",
    "pitch_res",
]


def _sample_windows(flagged=True):
    rows = []
    for i in range(5):
        rows.append(
            {
                "window_index": i,
                "session_id": 0,
                "t_start": float(i * 1.5),
                "t_end": float(i * 1.5 + 2.9),
                "anomaly_score": 5.0 if flagged and i in (1, 2) else 1.0,
                "threshold": 4.0,
                "is_anomaly": bool(flagged and i in (1, 2)),
                "feat_err_gyro_x": 0.5 if flagged and i in (1, 2) else 0.1,
            }
        )
    return pd.DataFrame(rows)


def test_build_report_validates():
    report = build_flight_report(
        flight_id="f1",
        analysis_status="completed",
        windows=_sample_windows(True),
        threshold=4.0,
        variant="conservative",
        features=FEATURES,
    )
    assert report.schema_version == "1.0"
    assert report.ai_analysis is None
    assert report.summary.total_windows == 5
    assert report.summary.flagged_windows == 2
    assert report.summary.flagged_rate == pytest.approx(0.4)
    assert report.summary.event_count == 1
    assert len(report.events) == 1
    validate_report(report)


def test_no_anomaly_status():
    report = build_flight_report(
        flight_id="f1",
        analysis_status="completed",
        windows=_sample_windows(False),
        threshold=4.0,
        variant="conservative",
        features=FEATURES,
    )
    assert report.analysis_status == "no_anomaly"
    assert report.summary.event_count == 0
    assert report.events == []


def test_zero_windows_null_rate():
    report = build_flight_report(
        flight_id="f1",
        analysis_status="insufficient_data",
        windows=pd.DataFrame(),
        threshold=4.0,
        variant="conservative",
        features=FEATURES,
        errors=["no windows"],
    )
    assert report.summary.total_windows == 0
    assert report.summary.flagged_rate is None


def test_bad_flagged_rate_rejected():
    with pytest.raises(ValidationError):
        FlightReport.model_validate(
            {
                "schema_version": "1.0",
                "flight_id": "x",
                "analysis_status": "completed",
                "detector": {
                    "model": "LSTMAE",
                    "variant": "conservative",
                    "feature_count": 12,
                    "window_length": 30,
                    "stride": 15,
                    "threshold": 4.0,
                    "features": FEATURES,
                },
                "summary": {
                    "total_windows": 10,
                    "flagged_windows": 1,
                    "flagged_rate": 0.5,  # wrong
                    "event_count": 0,
                },
                "events": [],
                "data_quality_warnings": [],
                "grouping_policy": {},
                "ai_analysis": None,
                "errors": [],
            }
        )


def test_event_duration_must_match_interval():
    with pytest.raises(ValidationError):
        FlightReport.model_validate(
            {
                "schema_version": "1.0",
                "flight_id": "x",
                "analysis_status": "completed",
                "detector": {
                    "model": "LSTMAE",
                    "variant": "conservative",
                    "feature_count": 12,
                    "window_length": 30,
                    "stride": 15,
                    "threshold": 4.0,
                    "features": FEATURES,
                },
                "summary": {
                    "total_windows": 1,
                    "flagged_windows": 1,
                    "flagged_rate": 1.0,
                    "event_count": 1,
                },
                "events": [
                    {
                        "event_id": "event-001",
                        "session_id": 0,
                        "start_time_s": 0.0,
                        "end_time_s": 3.0,
                        "duration_s": 9.0,  # wrong inflation
                        "peak_score": 5.0,
                        "mean_flagged_score": 5.0,
                        "flagged_window_count": 1,
                        "threshold": 4.0,
                        "window_indices": [0],
                        "affected_signals": [],
                        "evidence": [],
                        "data_quality_warnings": [],
                    }
                ],
                "data_quality_warnings": [],
                "grouping_policy": {},
                "ai_analysis": None,
                "errors": [],
            }
        )
