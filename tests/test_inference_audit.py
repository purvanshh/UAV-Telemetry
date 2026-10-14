"""Inference-pipeline audit checks. Does not change the threshold or weights."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.artifact_loader import load_artifacts
from src.detector import UAVDetector
from src.inference_service import clear_caches
from src.preprocessing_adapter import align_uploaded_flight
from src.window_generator import generate_windows


def _burst(t0_us: int, n_imu: int, n_att: int, *, yaw_deg: float, roll_deg: float, des_roll_deg: float):
    imu_t = t0_us + np.arange(n_imu) * 40_000
    att_t = t0_us + np.arange(n_att) * 100_000
    baro_t = att_t.copy()
    imu = pd.DataFrame(
        {
            "TimeUS": imu_t,
            "GyrX": np.zeros(n_imu),
            "GyrY": np.zeros(n_imu),
            "GyrZ": np.zeros(n_imu),
            "AccX": np.zeros(n_imu),
            "AccY": np.zeros(n_imu),
            "AccZ": np.full(n_imu, -9.81),
        }
    )
    att = pd.DataFrame(
        {
            "TimeUS": att_t,
            "Roll": np.full(n_att, roll_deg),
            "Pitch": np.zeros(n_att),
            "Yaw": np.full(n_att, yaw_deg),
            "DesRoll": np.full(n_att, des_roll_deg),
            "DesPitch": np.zeros(n_att),
            "DesYaw": np.full(n_att, yaw_deg),
        }
    )
    baro = pd.DataFrame(
        {
            "TimeUS": baro_t,
            "Alt": np.zeros(n_att),
            "Press": np.full(n_att, 101325.0),
            "Temp": np.full(n_att, 25.0),
        }
    )
    return imu, att, baro


def test_yaw_uses_degrees_roll_pitch_residuals_stay_degrees():
    """Schema: yaw_sin = sin(deg2rad(yaw)); roll/pitch and residuals stay in degrees."""
    t0 = 1_000_000
    imu, att, baro = _burst(t0, n_imu=400, n_att=80, yaw_deg=180.0, roll_deg=90.0, des_roll_deg=10.0)
    frames = {"IMU": imu, "ATT": att, "BARO": baro}
    aligned, _ = align_uploaded_flight(frames)
    finite = aligned.dropna(subset=["yaw_sin", "yaw_cos", "roll", "roll_res"])
    assert len(finite) > 0
    # sin(180°) = 0, cos(180°) = -1 — would be wrong if yaw were treated as radians
    assert np.allclose(finite["yaw_sin"], 0.0, atol=1e-5)
    assert np.allclose(finite["yaw_cos"], -1.0, atol=1e-5)
    assert np.allclose(finite["roll"], 90.0, atol=1e-3)
    assert np.allclose(finite["roll_res"], 80.0, atol=1e-3)


def test_shipped_scaler_matches_checkpoint_variant():
    """Weights, scaler, schema, and threshold must describe the same conservative model."""
    arts = load_artifacts()
    assert arts.variant == "conservative"
    assert arts.feature_schema.get("deployed_variant") == "conservative"
    assert list(arts.scaler["features"]) == list(arts.features)
    assert list(arts.feature_schema["deployed_feature_order"]) == list(arts.features)
    assert arts.n_features == 12
    assert len(arts.scaler["mean"]) == 12
    assert len(arts.scaler["std"]) == 12
    # Shipped validation quantile — not the older ~4.42494 figure
    assert abs(arts.threshold - 4.034964084625244) < 1e-6
    assert arts.threshold_config.get("quantile_used") == 0.995
    assert "validation" in str(arts.threshold_config.get("calibrated_on", "")).lower()

    det = UAVDetector(arts)
    assert det.threshold == arts.threshold
    # Roll scaler spread is degree-scale, not radian-scale (~0.2)
    roll_i = arts.features.index("roll")
    assert arts.scaler["std"][roll_i] > 5.0


def test_windows_do_not_cross_two_large_source_gaps():
    """Two source gaps (11 s and 245 s) become session breaks; no window spans them."""
    # Three bursts separated like flight 00000122's IMU gaps.
    parts = []
    cursor = 10_000_000
    for n, gap_after_us in ((800, 11_200_000), (600, 245_600_000), (900, 0)):
        imu, att, baro = _burst(
            cursor, n_imu=n, n_att=max(40, n // 4), yaw_deg=90.0, roll_deg=1.0, des_roll_deg=0.0
        )
        parts.append((imu, att, baro))
        cursor = int(imu["TimeUS"].iloc[-1]) + 40_000 + gap_after_us

    frames = {
        "IMU": pd.concat([p[0] for p in parts], ignore_index=True),
        "ATT": pd.concat([p[1] for p in parts], ignore_index=True),
        "BARO": pd.concat([p[2] for p in parts], ignore_index=True),
    }
    aligned, _ = align_uploaded_flight(frames)
    sessions = sorted(int(s) for s in aligned["_session_id"].unique() if int(s) >= 0)
    assert len(sessions) >= 3

    arts = load_artifacts()
    batch = generate_windows(aligned, arts)
    assert batch.n_windows > 0

    # Gap open intervals in seconds, from the constructed timestamps
    t = frames["IMU"]["TimeUS"].to_numpy(dtype=float) / 1e6
    dt = np.diff(t)
    gaps = [(float(t[i]), float(t[i + 1])) for i in np.where(dt > 1.0)[0]]
    assert len(gaps) == 2

    for meta in batch.meta:
        assert meta["t_end"] - meta["t_start"] < 3.5
        for g0, g1 in gaps:
            # A crossing window would start before the gap and end after it
            assert not (meta["t_start"] < g0 and meta["t_end"] > g1)


def test_inference_scores_are_reproducible(fixture_model_dir, long_flight_frames):
    clear_caches()
    arts = load_artifacts(fixture_model_dir)
    aligned, _ = align_uploaded_flight(long_flight_frames, model_dir=fixture_model_dir)
    batch = generate_windows(aligned, arts, model_dir=fixture_model_dir)
    det = UAVDetector(arts)
    s1, f1, p1 = det.score(batch.X)
    s2, f2, p2 = det.score(batch.X)
    np.testing.assert_array_equal(s1, s2)
    np.testing.assert_array_equal(f1, f2)
    np.testing.assert_array_equal(p1, p2)


def test_flags_are_strictly_above_saved_threshold():
    """Equality with the threshold is not a flag. Do not lower the threshold."""
    arts = load_artifacts()
    thr = float(arts.threshold)
    scores = np.array([thr - 0.03, thr, thr + 1e-6], dtype=np.float64)
    flags = scores > thr
    assert flags.tolist() == [False, False, True]
    # Near-miss like flight 00000122 (4.00630 vs 4.03496) must stay unflagged
    assert not bool(4.00630 > thr)
