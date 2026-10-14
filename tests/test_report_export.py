"""Export helpers."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.report_builder import build_flight_report
from src.report_export import (
    events_to_csv_bytes,
    report_to_json_bytes,
    windows_to_csv_bytes,
    write_report_json,
)


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


def test_json_roundtrip(tmp_path):
    windows = pd.DataFrame(
        [
            {
                "window_index": 0,
                "session_id": 0,
                "t_start": 0.0,
                "t_end": 2.9,
                "anomaly_score": 5.0,
                "threshold": 4.0,
                "is_anomaly": True,
            }
        ]
    )
    report = build_flight_report(
        flight_id="exp",
        analysis_status="completed",
        windows=windows,
        threshold=4.0,
        variant="conservative",
        features=FEATURES,
    )
    raw = report_to_json_bytes(report)
    data = json.loads(raw.decode("utf-8"))
    assert data["schema_version"] == "1.0"
    assert data["ai_analysis"] is None
    assert data["summary"]["flagged_windows"] == 1
    path = write_report_json(report, tmp_path / "r.json")
    assert path.exists()
    assert json.loads(path.read_text())["flight_id"] == "exp"


def test_csv_exports():
    windows = pd.DataFrame(
        [
            {
                "window_index": 0,
                "session_id": 0,
                "t_start": 0.0,
                "t_end": 2.9,
                "anomaly_score": 1.0,
                "threshold": 4.0,
                "is_anomaly": False,
            }
        ]
    )
    assert b"anomaly_score" in windows_to_csv_bytes(windows)
    empty_events = events_to_csv_bytes(None)
    assert b"event_id" in empty_events
