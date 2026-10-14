"""Evidence pack construction."""

from __future__ import annotations

import json

import pandas as pd

from src.evidence_builder import (
    DETECTOR_LIMITATIONS,
    build_evidence_pack,
    evidence_to_prompt_json,
)
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


def _report_with_events():
    windows = pd.DataFrame(
        [
            {
                "window_index": i,
                "session_id": 0,
                "t_start": float(i * 1.5),
                "t_end": float(i * 1.5 + 2.9),
                "anomaly_score": 5.0 if i in (2, 3) else 1.0,
                "threshold": 4.0,
                "is_anomaly": i in (2, 3),
                "feat_err_gyro_x": 1.2 if i in (2, 3) else 0.1,
            }
            for i in range(8)
        ]
    )
    return build_flight_report(
        flight_id="ev-1",
        analysis_status="completed",
        windows=windows,
        threshold=4.0,
        variant="conservative",
        features=FEATURES,
        data_quality_warnings=["BARO pressure exceeds typical surface range"],
    ), windows


def test_evidence_includes_detector_limits_and_features():
    report, windows = _report_with_events()
    pack = build_evidence_pack(report, windows=windows)
    assert pack["flight_id"] == "ev-1"
    assert pack["detector"]["feature_count"] == 12
    assert pack["detector"]["features"] == FEATURES
    assert "alt" not in pack["detector"]["features"]
    assert pack["data_quality_warnings"]
    assert DETECTOR_LIMITATIONS[0] in pack["detector_limitations"]
    assert pack["events"]
    assert "baseline_windows" in pack["events"][0]
    # Serializes
    text = evidence_to_prompt_json(pack)
    assert "gyro_x" in text
    json.loads(text)


def test_evidence_respects_selected_event():
    report, windows = _report_with_events()
    eid = report.events[0].event_id
    pack = build_evidence_pack(
        report, windows=windows, selected_event_ids=[eid]
    )
    assert len(pack["events"]) == 1
    assert pack["events"][0]["event_id"] == eid


def test_evidence_char_budget_truncates():
    report, windows = _report_with_events()
    full = build_evidence_pack(report, windows=windows, max_chars=100_000)
    pack = build_evidence_pack(report, windows=windows, max_chars=600)
    assert pack["_approx_chars"] <= 600
    assert pack["_approx_chars"] < full["_approx_chars"]
    assert pack.get("context_truncated") is True or len(pack["events"]) < len(
        full["events"]
    )


def test_zero_event_evidence_includes_score_context_not_fake_events():
    windows = pd.DataFrame(
        [
            {
                "window_index": i,
                "session_id": 0,
                "t_start": float(i * 1.5),
                "t_end": float(i * 1.5 + 2.9),
                "anomaly_score": 0.001 + i * 0.0005,
                "threshold": 4.03496,
                "is_anomaly": False,
                "feat_err_gyro_x": 0.0001 * (i + 1),
                "feat_err_acc_z": 0.0002,
            }
            for i in range(6)
        ]
    )
    report = build_flight_report(
        flight_id="no-anom",
        analysis_status="no_anomaly",
        windows=windows,
        threshold=4.03496,
        variant="conservative",
        features=FEATURES,
        data_quality_warnings=[
            "BARO pressure exceeds typical surface range. "
            "The deployed LSTM excludes pressure."
        ],
        score_mean=0.002,
        score_p50=0.002,
        score_p95=0.003,
        score_max=0.00308,
    )
    assert report.summary.event_count == 0
    pack = build_evidence_pack(report, windows=windows)
    assert pack["events"] == []
    assert pack["score_context"]["no_threshold_exceedances"] is True
    assert pack["score_context"]["score_distribution"]["max"] is not None
    assert pack["top_windows_by_score"]
    assert all(w["below_threshold"] for w in pack["top_windows_by_score"])
    assert all(
        w["label"] == "unflagged_high_score_context_only"
        for w in pack["top_windows_by_score"]
    )
    assert "alt" not in pack["detector"]["features"]
    assert pack["data_quality_warnings"]
    text = evidence_to_prompt_json(pack)
    assert "score_context" in text
    assert "Do not claim BARO" in " ".join(pack["instructions_for_model"]) or any(
        "BARO" in x for x in pack["detector_limitations"]
    )
