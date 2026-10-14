"""Event grouping policy tests."""

from __future__ import annotations

import pandas as pd
import pytest

from src.event_builder import build_events, events_to_dataframe


def _windows(rows):
    return pd.DataFrame(rows)


def test_no_flagged_returns_empty():
    df = _windows(
        [
            {
                "window_index": 0,
                "session_id": 0,
                "t_start": 0.0,
                "t_end": 2.9,
                "anomaly_score": 1.0,
                "is_anomaly": False,
            }
        ]
    )
    assert build_events(df) == []


def test_overlapping_windows_merge():
    # Overlapping 3 s windows with stride 1.5
    df = _windows(
        [
            {
                "window_index": 0,
                "session_id": 0,
                "t_start": 10.0,
                "t_end": 12.9,
                "anomaly_score": 5.0,
                "is_anomaly": True,
            },
            {
                "window_index": 1,
                "session_id": 0,
                "t_start": 11.5,
                "t_end": 14.4,
                "anomaly_score": 6.0,
                "is_anomaly": True,
            },
        ]
    )
    events = build_events(df, max_gap_s=1.5)
    assert len(events) == 1
    assert events[0].start_time_s == 10.0
    assert events[0].end_time_s == 14.4
    assert events[0].duration_s == pytest.approx(4.4)
    assert events[0].flagged_window_count == 2
    assert events[0].peak_score == 6.0
    # Duration is NOT 2 * 3.0
    assert events[0].duration_s != pytest.approx(6.0)


def test_gap_within_max_merges():
    df = _windows(
        [
            {
                "window_index": 0,
                "session_id": 0,
                "t_start": 0.0,
                "t_end": 3.0,
                "anomaly_score": 5.0,
                "is_anomaly": True,
            },
            {
                "window_index": 1,
                "session_id": 0,
                "t_start": 4.0,  # gap 1.0 s after previous end
                "t_end": 7.0,
                "anomaly_score": 5.5,
                "is_anomaly": True,
            },
        ]
    )
    assert len(build_events(df, max_gap_s=1.5)) == 1
    assert len(build_events(df, max_gap_s=0.5)) == 2


def test_never_merge_across_sessions():
    df = _windows(
        [
            {
                "window_index": 0,
                "session_id": 0,
                "t_start": 0.0,
                "t_end": 3.0,
                "anomaly_score": 5.0,
                "is_anomaly": True,
            },
            {
                "window_index": 1,
                "session_id": 1,
                "t_start": 3.1,
                "t_end": 6.1,
                "anomaly_score": 5.0,
                "is_anomaly": True,
            },
        ]
    )
    events = build_events(df, max_gap_s=10.0)
    assert len(events) == 2
    assert {e.session_id for e in events} == {0, 1}


def test_isolated_spikes_separate():
    df = _windows(
        [
            {
                "window_index": 0,
                "session_id": 0,
                "t_start": 0.0,
                "t_end": 3.0,
                "anomaly_score": 5.0,
                "is_anomaly": True,
            },
            {
                "window_index": 1,
                "session_id": 0,
                "t_start": 20.0,
                "t_end": 23.0,
                "anomaly_score": 5.0,
                "is_anomaly": True,
            },
        ]
    )
    events = build_events(df, max_gap_s=1.5)
    assert len(events) == 2


def test_events_dataframe():
    df = _windows(
        [
            {
                "window_index": 0,
                "session_id": 0,
                "t_start": 1.0,
                "t_end": 4.0,
                "anomaly_score": 5.0,
                "is_anomaly": True,
            }
        ]
    )
    events = build_events(df)
    edf = events_to_dataframe(events, threshold=4.0)
    assert len(edf) == 1
    assert edf.iloc[0]["peak_over_threshold"] == pytest.approx(1.0)
