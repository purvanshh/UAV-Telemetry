"""Event metric helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from src.event_builder import AnomalyEvent
from src.event_metrics import (
    enrich_event_dict,
    threshold_relative_score,
    top_affected_signals,
)


def test_threshold_relative():
    assert threshold_relative_score(8.0, 4.0) == pytest.approx(2.0)
    assert threshold_relative_score(5.0, 0.0) is None
    assert threshold_relative_score(float("nan"), 4.0) is None


def test_top_affected_empty_without_evidence():
    ev = AnomalyEvent(
        event_id="event-001",
        session_id=0,
        start_time_s=0.0,
        end_time_s=3.0,
        window_indices=[0],
        scores=[5.0],
    )
    assert top_affected_signals(ev) == []


def test_top_affected_ranked():
    ev = AnomalyEvent(
        event_id="event-001",
        session_id=0,
        start_time_s=0.0,
        end_time_s=3.0,
        window_indices=[0],
        scores=[5.0],
        per_feature_peak={"gyro_x": 0.1, "acc_z": 2.0, "roll": 0.5},
        per_feature_mean={"gyro_x": 0.1, "acc_z": 1.5, "roll": 0.4},
    )
    top = top_affected_signals(ev, top_k=2)
    assert [t["signal"] for t in top] == ["acc_z", "roll"]


def test_enrich_includes_duration_not_inflated():
    ev = AnomalyEvent(
        event_id="event-001",
        session_id=0,
        start_time_s=10.0,
        end_time_s=14.4,
        window_indices=[0, 1],
        scores=[5.0, 6.0],
        per_feature_peak={"gyro_x": 1.2},
        per_feature_mean={"gyro_x": 1.0},
    )
    aligned = pd.DataFrame(
        {
            "_t_s": [10.0, 11.0, 12.0, 13.0, 14.0],
            "gyro_x": [0.0, 0.1, 0.2, 0.1, 0.0],
        }
    )
    payload = enrich_event_dict(
        ev, threshold=4.0, features=["gyro_x"], aligned=aligned
    )
    assert payload["duration_s"] == pytest.approx(4.4)
    assert payload["affected_signals"] == ["gyro_x"]
    assert any(e["type"] == "per_feature_recon_error" for e in payload["evidence"])
    assert any(e["type"] == "local_telemetry_summary" for e in payload["evidence"])
