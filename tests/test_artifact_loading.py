"""Artifact loader consistency tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.artifact_loader import ArtifactError, load_artifacts, missing_artifacts
from src.inference_service import clear_caches


def test_fixture_artifacts_load(fixture_model_dir):
    clear_caches()
    arts = load_artifacts(fixture_model_dir)
    assert arts.n_features == 12
    assert len(arts.features) == 12
    assert arts.win_samples == 30
    assert arts.stride_samples == 15
    assert arts.features == arts.scaler["features"]
    assert arts.threshold == pytest.approx(4.034964084625244)
    assert arts.variant == "conservative"


def test_feature_order_is_twelve_conservative(fixture_model_dir):
    arts = load_artifacts(fixture_model_dir)
    assert "alt" not in arts.features
    assert "press" not in arts.features
    assert "temp" not in arts.features
    assert arts.features[0] == "gyro_x"
    assert arts.features[-1] == "pitch_res"


def test_missing_checkpoint_reported(fixture_model_dir, tmp_path):
    # Copy configs but omit checkpoint
    dest = tmp_path / "partial"
    dest.mkdir()
    for name in (
        "model_config.json",
        "feature_schema.json",
        "scaler.json",
        "threshold_config.json",
        "preprocessing.py",
    ):
        (dest / name).write_bytes((fixture_model_dir / name).read_bytes())
    miss = missing_artifacts(dest)
    assert "lstm_autoencoder.pth" in miss
    with pytest.raises(ArtifactError, match="lstm_autoencoder"):
        load_artifacts(dest)


def test_feature_mismatch_raises(fixture_model_dir, tmp_path):
    import shutil

    dest = tmp_path / "bad"
    shutil.copytree(fixture_model_dir, dest)
    cfg = json.loads((dest / "model_config.json").read_text())
    cfg["features"] = cfg["features"][:-1] + ["alt"]  # still 12 but wrong last
    (dest / "model_config.json").write_text(json.dumps(cfg))
    with pytest.raises(ArtifactError, match="Feature order mismatch"):
        load_artifacts(dest)


def test_project_uav_model_missing_weights():
    """Production uav_model/ may lack .pth until the user copies Kaggle export."""
    root = Path(__file__).resolve().parents[1] / "uav_model"
    miss = missing_artifacts(root)
    # preprocessing + configs should exist; weights/scaler may be missing
    assert "preprocessing.py" not in miss
    assert "model_config.json" not in miss
