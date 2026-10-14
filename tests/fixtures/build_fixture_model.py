"""Build a temporary uav_model fixture with a random-init LSTMAE checkpoint.

Used only for automated tests. Not a trained detector — scores are meaningless
except for shape / determinism / pipeline wiring checks.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC_MODEL = ROOT / "uav_model"
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


def build_fixture_model(dest: Path, seed: int = 0) -> Path:
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    # Copy preprocessing from project uav_model
    shutil.copy(SRC_MODEL / "preprocessing.py", dest / "preprocessing.py")

    model_config = {
        "architecture": "LSTMAE",
        "n_features": 12,
        "hidden_dim": 48,
        "latent_dim": 12,
        "num_layers": 1,
        "win_samples": 30,
        "stride_samples": 15,
        "features": FEATURES,
        "device_at_inference": "cpu",
        "fixture": True,
    }
    feature_schema = {
        "version": "0.3.0-fixture",
        "model_feature_order": FEATURES
        + ["alt", "press", "temp", "alt_rate", "press_rate"],
        "deployed_feature_order": FEATURES,
        "deployed_variant": "conservative",
        "alignment": {
            "target_hz": 10.0,
            "max_gap_s": 0.5,
            "session_gap_s": 1.0,
        },
        "window": {
            "win_samples": 30,
            "stride_samples": 15,
            "win_seconds": 3.0,
            "stride_seconds": 1.5,
        },
    }
    # Identity-ish scaler
    scaler = {
        "mean": [0.0] * 12,
        "std": [1.0] * 12,
        "features": FEATURES,
        "fit_on_flights": ["fixture"],
    }
    threshold_config = {
        "variant": "conservative",
        "lstm_ae": 4.034964084625244,
        "iforest": 0.46,
        "quantile_used": 0.995,
        "calibrated_on": "fixture",
    }

    for name, obj in (
        ("model_config.json", model_config),
        ("feature_schema.json", feature_schema),
        ("scaler.json", scaler),
        ("threshold_config.json", threshold_config),
    ):
        with open(dest / name, "w") as f:
            json.dump(obj, f, indent=2)

    # Import architecture from src
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.detector import LSTMAE

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = LSTMAE(12, 48, 12, 30, 1)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_features": 12,
            "hidden_dim": 48,
            "latent_dim": 12,
            "win_samples": 30,
            "num_layers": 1,
        },
        dest / "lstm_autoencoder.pth",
    )
    return dest


if __name__ == "__main__":
    out = Path(__file__).parent / "uav_model"
    build_fixture_model(out)
    print("wrote", out)
