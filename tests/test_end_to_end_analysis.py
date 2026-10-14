"""End-to-end analyze_flight orchestration."""

from __future__ import annotations

import pytest

from src.inference_service import analyze_flight, clear_caches


def test_analyze_long_flight(fixture_model_dir, long_flight_frames):
    clear_caches()
    result = analyze_flight(
        long_flight_frames,
        flight_id="test-long",
        model_dir=str(fixture_model_dir),
    )
    assert result.ok
    assert result.status == "completed"
    assert result.n_windows > 0
    assert result.windows is not None
    for col in (
        "flight_id",
        "window_index",
        "t_start",
        "t_end",
        "anomaly_score",
        "threshold",
        "is_anomaly",
    ):
        assert col in result.windows.columns
    assert result.threshold == pytest.approx(4.034964084625244)
    assert len(result.features) == 12
    assert "alt" not in result.features
    # Repeated call is deterministic
    result2 = analyze_flight(
        long_flight_frames,
        flight_id="test-long",
        model_dir=str(fixture_model_dir),
    )
    assert result2.n_windows == result.n_windows
    assert result2.n_flagged == result.n_flagged
    assert abs(result2.score_mean - result.score_mean) < 1e-7


def test_analyze_short_flight_insufficient(fixture_model_dir, sample_frames):
    clear_caches()
    result = analyze_flight(
        sample_frames,
        flight_id="short",
        model_dir=str(fixture_model_dir),
    )
    assert not result.ok
    assert result.status == "insufficient_data"
    assert result.n_windows == 0


def test_analyze_missing_artifacts(tmp_path, long_flight_frames):
    clear_caches()
    result = analyze_flight(
        long_flight_frames,
        flight_id="x",
        model_dir=str(tmp_path),
    )
    assert not result.ok
    assert result.status == "error"
    assert any("Missing" in e or "not found" in e.lower() for e in result.errors)


def test_no_healthy_label_when_unflagged(fixture_model_dir, long_flight_frames):
    """Even with zero flags, result must not claim the flight is healthy."""
    clear_caches()
    result = analyze_flight(
        long_flight_frames,
        flight_id="maybe-clean",
        model_dir=str(fixture_model_dir),
    )
    assert result.ok
    # Fixture random model may or may not flag; if none flagged, warning required
    if result.n_flagged == 0:
        assert any("not a guarantee" in w.lower() for w in result.warnings)
