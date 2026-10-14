"""Diagnostic pipeline trace helpers."""

from __future__ import annotations

import pandas as pd

from src.diagnostic_trace import build_diagnostic_trace, format_diagnostic_trace
from src.evidence_builder import build_evidence_pack
from src.report_builder import build_flight_report

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


def test_trace_for_zero_event_report():
    windows = pd.DataFrame(
        [
            {
                "window_index": i,
                "session_id": 0,
                "t_start": float(i),
                "t_end": float(i + 2.9),
                "anomaly_score": 0.001 * (i + 1),
                "threshold": 4.0,
                "is_anomaly": False,
            }
            for i in range(5)
        ]
    )
    report = build_flight_report(
        flight_id="t1",
        analysis_status="no_anomaly",
        windows=windows,
        threshold=4.0,
        variant="conservative",
        features=FEATURES,
        score_max=0.005,
    )
    pack = build_evidence_pack(report, windows=windows)
    trace = build_diagnostic_trace(report, windows=windows, evidence_pack=pack)
    assert trace["scored_windows"] == 5
    assert trace["flagged_windows"] == 0
    assert trace["event_count"] == 0
    assert trace["score_distribution"]["max"] == 0.005
    assert trace["evidence_events_sent"] == 0
    assert trace["evidence_top_windows_sent"] == 5
    text = format_diagnostic_trace(trace)
    assert "flagged=0" in text
    assert "OPENAI" not in text
    assert "sk-" not in text
