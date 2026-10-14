"""Detector scoring — shape, threshold flags, determinism."""

from __future__ import annotations

import numpy as np
import pytest

from src.artifact_loader import load_artifacts
from src.detector import UAVDetector
from src.inference_service import clear_caches
from src.preprocessing_adapter import align_uploaded_flight
from src.window_generator import generate_windows


def test_score_shape_and_flags(fixture_model_dir, long_flight_frames):
    clear_caches()
    arts = load_artifacts(fixture_model_dir)
    aligned, _ = align_uploaded_flight(long_flight_frames, model_dir=fixture_model_dir)
    batch = generate_windows(aligned, arts, model_dir=fixture_model_dir)
    det = UAVDetector(arts)
    scores, flags, per_feat = det.score(batch.X)
    assert scores.shape == (batch.n_windows,)
    assert flags.shape == (batch.n_windows,)
    assert per_feat.shape == (batch.n_windows, 12)
    assert flags.dtype == bool
    # flags must equal score > saved threshold (not recalibrated)
    np.testing.assert_array_equal(flags, scores > arts.threshold)


def test_inference_deterministic(fixture_model_dir, long_flight_frames):
    arts = load_artifacts(fixture_model_dir)
    aligned, _ = align_uploaded_flight(long_flight_frames, model_dir=fixture_model_dir)
    batch = generate_windows(aligned, arts, model_dir=fixture_model_dir)
    det = UAVDetector(arts)
    s1 = det.recon_error_per_window(batch.X)
    s2 = det.recon_error_per_window(batch.X)
    np.testing.assert_allclose(s1, s2, rtol=0, atol=1e-7)


def test_scaler_applied(fixture_model_dir, long_flight_frames):
    arts = load_artifacts(fixture_model_dir)
    aligned, _ = align_uploaded_flight(long_flight_frames, model_dir=fixture_model_dir)
    batch = generate_windows(aligned, arts, model_dir=fixture_model_dir)
    det = UAVDetector(arts)
    Xn = det.scale(batch.X)
    # fixture scaler is mean 0 std 1 → unchanged
    np.testing.assert_allclose(Xn, batch.X, rtol=1e-5, atol=1e-5)


def test_checkpoint_feature_count(fixture_model_dir):
    arts = load_artifacts(fixture_model_dir)
    det = UAVDetector(arts)
    # smoke forward
    import torch

    x = torch.zeros(2, 30, 12)
    with torch.no_grad():
        y = det.model(x)
    assert y.shape == (2, 30, 12)
