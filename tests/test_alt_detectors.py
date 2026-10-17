"""Window sets, reference fitting and burden matching."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.alt_detectors import (
    DETECTORS, Reference, build_windowset, components, fit_reference, match_threshold, window_indices,
)
from src.artifact_loader import load_artifacts
from src.context_features import derive_signals
from src.detector import UAVDetector
from src.preprocessing_adapter import align_uploaded_flight_detailed
from src.window_generator import generate_windows


def _prep(frames, model_dir):
    arts = load_artifacts(model_dir)
    det = UAVDetector(arts)
    d = align_uploaded_flight_detailed(frames, model_dir)
    sig = derive_signals(d.aligned, d.clean_streams["ATT"], d.prep.gap_aware_interp)
    mu = np.asarray(arts.scaler["mean"], dtype=np.float32)
    sd = np.asarray(arts.scaler["std"], dtype=np.float32)
    return arts, det, d, sig, mu, sd


def test_windows_match_deployed_generation_exactly(fixture_model_dir, gapped_flight_frames):
    arts, det, d, sig, mu, sd = _prep(gapped_flight_frames, fixture_model_dir)
    batch = generate_windows(d.aligned, arts, model_dir=fixture_model_dir)
    ws = build_windowset(d.aligned, sig, arts.features, mu, sd, 30, 15, det)
    assert ws.n == batch.n_windows
    np.testing.assert_allclose(ws.meta["t_start"], [m["t_start"] for m in batch.meta])
    np.testing.assert_array_equal(ws.meta["session_id"], [m["session_id"] for m in batch.meta])
    scores, _, per = det.score(batch.X)
    np.testing.assert_allclose(ws.lstm, scores, rtol=1e-5)
    np.testing.assert_allclose(ws.lstm_feat, per, rtol=1e-5)


def test_short_windows_never_cross_sessions(fixture_model_dir, gapped_flight_frames):
    arts, det, d, sig, mu, sd = _prep(gapped_flight_frames, fixture_model_dir)
    W = window_indices(d.aligned, arts.features, 10, 5)
    sid = d.aligned["_session_id"].to_numpy()
    assert len(W) and all(len(set(sid[w])) == 1 and sid[w[0]] >= 0 for w in W)


def test_reference_uses_only_given_windows_and_detectors_score(fixture_model_dir, long_flight_frames):
    arts, det, d, sig, mu, sd = _prep(long_flight_frames, fixture_model_dir)
    s30 = build_windowset(d.aligned, sig, arts.features, mu, sd, 30, 15, det)
    s10 = build_windowset(d.aligned, sig, arts.features, mu, sd, 10, 5)
    ref = fit_reference({30: [s30], 10: [s10]})
    assert ref.n_windows == {30: s30.n, 10: s10.n}
    for dd in DETECTORS:
        sc = dd.score(s30 if dd.win == 30 else s10, ref)
        assert sc.shape == (s30.n if dd.win == 30 else s10.n,)
    # under-sampled phases fall back to the global quantile (documented behaviour)
    assert set(ref.phase_q["lstm"]) == {"steady", "active", "vertical"}


def test_flatline_component_flags_constant_channel(fixture_model_dir, long_flight_frames):
    arts, det, d, sig, mu, sd = _prep(long_flight_frames, fixture_model_dir)
    ref = fit_reference({30: [build_windowset(d.aligned, sig, arts.features, mu, sd, 30, 15, det)]})
    a = d.aligned.copy()
    a.loc[40:80, "gyro_x"] = 0.0
    s = build_windowset(a, sig, arts.features, mu, sd, 30, 15, det)
    base = build_windowset(d.aligned, sig, arts.features, mu, sd, 30, 15, det)
    assert components(s, ref)["flat"].max() > 10 * components(base, ref)["flat"].max()


def test_match_threshold_hits_target_burden():
    x = np.random.default_rng(0).normal(size=1000)
    for frac in (0.01, 0.05, 0.2):
        tau = match_threshold(x, frac)
        assert (x > tau).sum() == int(round(frac * 1000))
    assert np.isnan(match_threshold(np.array([np.nan]), 0.1))
    assert (x > match_threshold(x, 0.0)).sum() == 0
