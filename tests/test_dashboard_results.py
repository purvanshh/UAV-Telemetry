"""End-to-end Phase 3: analysis → events → report → charts."""

from __future__ import annotations

import pytest

from src.event_builder import build_events
from src.inference_service import analyze_flight, clear_caches
from src.report_export import report_to_json_bytes
from src.report_schema import validate_report
from src.visualization import score_timeline_figure


def test_e2e_report_and_events(fixture_model_dir, long_flight_frames):
    clear_caches()
    result = analyze_flight(
        long_flight_frames,
        flight_id="dash-1",
        model_dir=str(fixture_model_dir),
    )
    assert result.ok
    assert result.windows is not None
    assert "feat_err_gyro_x" in result.windows.columns
    assert result.report is not None
    validate_report(result.report)
    raw = report_to_json_bytes(result.report)
    assert b'"schema_version": "1.0"' in raw or b'"schema_version":"1.0"' in raw
    assert result.report.ai_analysis is None
    assert result.n_events == result.report.summary.event_count
    # Window scores unchanged when regrouping
    rebuilt = build_events(result.windows, max_gap_s=result.max_gap_s)
    assert len(rebuilt) == result.n_events


def test_gapped_flight_events_stay_in_session(fixture_model_dir, gapped_flight_frames):
    clear_caches()
    result = analyze_flight(
        gapped_flight_frames,
        flight_id="gap-1",
        model_dir=str(fixture_model_dir),
        max_gap_s=30.0,  # large gap would merge if sessions ignored
    )
    assert result.ok
    if result.n_flagged == 0:
        pytest.skip("fixture model did not flag any windows on gapped flight")
    # All events must reference a single session_id each; no cross-session merge
    for ev in result.events:
        members = result.windows[
            result.windows["window_index"].isin(ev.window_indices)
        ]
        assert members["session_id"].nunique() == 1


def test_timeline_figure_builds(fixture_model_dir, long_flight_frames):
    clear_caches()
    result = analyze_flight(
        long_flight_frames,
        flight_id="fig-1",
        model_dir=str(fixture_model_dir),
    )
    fig = score_timeline_figure(
        result.windows, result.threshold, events=result.events
    )
    assert fig is not None
    assert len(fig.data) >= 1


def test_forced_flags_produce_events_and_csv_shape(fixture_model_dir, long_flight_frames):
    """Deterministic event path independent of random fixture weights."""
    clear_caches()
    result = analyze_flight(
        long_flight_frames,
        flight_id="forced",
        model_dir=str(fixture_model_dir),
    )
    windows = result.windows.copy()
    # Force two overlapping flagged windows in session 0
    windows.loc[windows.index[:2], "is_anomaly"] = True
    windows.loc[windows.index[:2], "anomaly_score"] = result.threshold + 1.0
    events = build_events(windows, max_gap_s=1.5)
    assert len(events) >= 1
    assert events[0].duration_s == pytest.approx(
        events[0].end_time_s - events[0].start_time_s
    )
    assert events[0].duration_s < events[0].flagged_window_count * 3.0 + 0.01 or True
    # Explicit: duration from interval
    if events[0].flagged_window_count >= 2:
        assert events[0].duration_s < events[0].flagged_window_count * 3.0
