"""Window generation — length, stride, sessions, BARO-NaN policy."""

from __future__ import annotations

import numpy as np
import pytest

from src.artifact_loader import load_artifacts
from src.inference_service import clear_caches
from src.preprocessing_adapter import align_uploaded_flight
from src.window_generator import generate_windows


def test_window_shape_and_stride(fixture_model_dir, long_flight_frames):
    clear_caches()
    arts = load_artifacts(fixture_model_dir)
    aligned, _ = align_uploaded_flight(long_flight_frames, model_dir=fixture_model_dir)
    batch = generate_windows(aligned, arts, model_dir=fixture_model_dir)
    assert batch.n_windows > 0
    assert batch.X.shape[1] == 30
    assert batch.X.shape[2] == 12
    assert np.isfinite(batch.X).all()
    # consecutive window starts should differ by stride * dt (1.5 s at 10 Hz)
    if batch.n_windows >= 2:
        dt = batch.meta[1]["t_start"] - batch.meta[0]["t_start"]
        assert abs(dt - 1.5) < 0.05


def test_windows_do_not_cross_session_gap(fixture_model_dir, gapped_flight_frames):
    arts = load_artifacts(fixture_model_dir)
    aligned, _ = align_uploaded_flight(gapped_flight_frames, model_dir=fixture_model_dir)
    sids = set(int(s) for s in aligned["_session_id"].unique() if s >= 0)
    assert len(sids) >= 2

    batch = generate_windows(aligned, arts, model_dir=fixture_model_dir)
    assert batch.n_windows > 0
    for m in batch.meta:
        assert m["session_id"] >= 0
        # window duration should be ~2.9 s (30 samples at 10 Hz)
        assert (m["t_end"] - m["t_start"]) < 3.5


def test_conservative_keeps_windows_with_baro_nan_only(
    fixture_model_dir, long_flight_frames
):
    """NaN only in excluded BARO cols must not discard conservative windows."""
    arts = load_artifacts(fixture_model_dir)
    aligned, prep = align_uploaded_flight(
        long_flight_frames, model_dir=fixture_model_dir
    )
    # Inject NaN into alt/press at the start of each session (first-row rate artifact
    # already exists; force more BARO NaNs across the grid)
    aligned = aligned.copy()
    aligned.loc[:, "alt"] = np.nan
    aligned.loc[:, "press"] = np.nan
    aligned.loc[:, "temp"] = np.nan
    aligned.loc[:, "alt_rate"] = np.nan
    aligned.loc[:, "press_rate"] = np.nan

    # Full 17-feature filter would drop everything; conservative must keep some
    X_full, _ = prep.make_windows(
        aligned,
        arts.feature_schema["model_feature_order"],
        win_len=30,
        stride=15,
    )
    batch = generate_windows(aligned, arts, model_dir=fixture_model_dir)
    assert len(X_full) == 0
    assert batch.n_windows > 0


def test_short_flight_zero_windows(fixture_model_dir, sample_frames):
    arts = load_artifacts(fixture_model_dir)
    aligned, _ = align_uploaded_flight(sample_frames, model_dir=fixture_model_dir)
    batch = generate_windows(aligned, arts, model_dir=fixture_model_dir)
    # ~2 s of data < 3 s window
    assert batch.n_windows == 0
