"""Load and verify UAV model artifacts from ``uav_model/``."""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "uav_model"

CHECKPOINT_NAME = "lstm_autoencoder.pth"
CHECKPOINT_DIR_NAME = "lstm_autoencoder"

REQUIRED_FILES = (
    "model_config.json",
    "feature_schema.json",
    "scaler.json",
    "threshold_config.json",
    CHECKPOINT_NAME,
    "preprocessing.py",
)


class ArtifactError(RuntimeError):
    """Raised when model artifacts are missing or inconsistent."""


@dataclass
class ModelArtifacts:
    model_dir: Path
    model_config: Dict[str, Any]
    feature_schema: Dict[str, Any]
    scaler: Dict[str, Any]
    threshold_config: Dict[str, Any]
    checkpoint_path: Path
    features: List[str]
    win_samples: int
    stride_samples: int
    threshold: float
    hidden_dim: int
    latent_dim: int
    num_layers: int
    n_features: int
    warnings: List[str] = field(default_factory=list)

    @property
    def variant(self) -> str:
        return str(
            self.threshold_config.get("variant")
            or self.feature_schema.get("deployed_variant")
            or "conservative"
        )


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def resolve_model_dir(model_dir: Optional[Path | str] = None) -> Path:
    if model_dir is None:
        import os

        env = os.environ.get("UAV_MODEL_DIR")
        return Path(env) if env else DEFAULT_MODEL_DIR
    return Path(model_dir)


def _extracted_checkpoint_dir(model_dir: Path) -> Optional[Path]:
    """Return extracted Torch zip folder if present (macOS often unpacks .pth)."""
    cand = model_dir / CHECKPOINT_DIR_NAME
    if cand.is_dir() and (cand / "data.pkl").is_file() and (cand / "version").is_file():
        return cand
    return None


def _checkpoint_zip_looks_valid(path: Path) -> bool:
    """Torch zip checkpoints store records under ``<stem>/version``."""
    if not path.is_file() or path.stat().st_size < 64:
        return False
    stem = path.stem
    try:
        with zipfile.ZipFile(path, "r") as zf:
            names = set(zf.namelist())
        return f"{stem}/version" in names and f"{stem}/data.pkl" in names
    except zipfile.BadZipFile:
        return False


def ensure_checkpoint(model_dir: Optional[Path | str] = None) -> Path:
    """Ensure ``lstm_autoencoder.pth`` exists as a Torch-readable zip.

    If only the extracted ``lstm_autoencoder/`` folder is present (or the
    ``.pth`` was re-zipped without the required archive prefix), rebuild it.
    """
    d = resolve_model_dir(model_dir)
    pth = d / CHECKPOINT_NAME
    if _checkpoint_zip_looks_valid(pth):
        return pth

    extracted = _extracted_checkpoint_dir(d)
    if extracted is None:
        if pth.exists():
            raise ArtifactError(
                f"{CHECKPOINT_NAME} exists but is not a valid Torch checkpoint zip. "
                f"Re-export from Kaggle or place the extracted {CHECKPOINT_DIR_NAME}/ "
                "folder beside it so it can be repaired."
            )
        raise ArtifactError(
            f"Missing {CHECKPOINT_NAME}. Place the Kaggle export in {d} "
            "(see uav_model/ARTIFACTS.md)."
        )

    # Torch expects zip entries prefixed with the archive stem.
    prefix = Path(CHECKPOINT_NAME).stem
    if pth.exists():
        pth.unlink()
    with zipfile.ZipFile(pth, "w", compression=zipfile.ZIP_STORED) as zf:
        for fp in sorted(extracted.rglob("*")):
            if fp.is_file():
                arcname = f"{prefix}/{fp.relative_to(extracted).as_posix()}"
                zf.write(fp, arcname)

    if not _checkpoint_zip_looks_valid(pth):
        raise ArtifactError(
            f"Failed to rebuild a valid {CHECKPOINT_NAME} from {extracted}."
        )
    return pth


def missing_artifacts(model_dir: Optional[Path | str] = None) -> List[str]:
    d = resolve_model_dir(model_dir)
    missing = [name for name in REQUIRED_FILES if not (d / name).exists()]
    # Treat a valid extracted checkpoint folder as satisfying the .pth requirement
    if CHECKPOINT_NAME in missing and _extracted_checkpoint_dir(d) is not None:
        missing = [m for m in missing if m != CHECKPOINT_NAME]
    return missing


def load_artifacts(model_dir: Optional[Path | str] = None) -> ModelArtifacts:
    """Load artifacts and assert cross-file consistency.

    Does not load torch weights into memory — only verifies the checkpoint
    file exists and that JSON configs agree on feature count / order.
    """
    d = resolve_model_dir(model_dir)
    if not d.is_dir():
        raise ArtifactError(f"Model directory not found: {d}")

    # Repair / materialize .pth before the missing-file check when possible
    try:
        ensure_checkpoint(d)
    except ArtifactError:
        pass

    missing = missing_artifacts(d)
    if missing:
        raise ArtifactError(
            "Missing required model artifact(s): "
            + ", ".join(missing)
            + f". Place your Kaggle uav_model export in {d} "
            "(see uav_model/ARTIFACTS.md)."
        )

    checkpoint_path = ensure_checkpoint(d)

    model_config = _read_json(d / "model_config.json")
    feature_schema = _read_json(d / "feature_schema.json")
    scaler = _read_json(d / "scaler.json")
    threshold_config = _read_json(d / "threshold_config.json")
    warnings: List[str] = []

    features = list(
        model_config.get("features")
        or feature_schema.get("deployed_feature_order")
        or []
    )
    if not features:
        raise ArtifactError("No deployed feature list found in artifacts.")

    schema_feats = feature_schema.get("deployed_feature_order")
    if schema_feats and list(schema_feats) != features:
        raise ArtifactError(
            "Feature order mismatch between model_config.json and "
            "feature_schema.json deployed_feature_order."
        )

    scaler_feats = scaler.get("features")
    if scaler_feats and list(scaler_feats) != features:
        raise ArtifactError(
            "Feature order mismatch between scaler.json and model_config.json."
        )

    n_features = int(model_config.get("n_features", len(features)))
    if n_features != len(features):
        raise ArtifactError(
            f"n_features={n_features} does not match feature list length "
            f"{len(features)}."
        )
    if len(features) != 12:
        raise ArtifactError(
            f"Deployed model must use 12 features; found {len(features)}."
        )

    if "mean" not in scaler or "std" not in scaler:
        raise ArtifactError("scaler.json must contain 'mean' and 'std'.")
    if len(scaler["mean"]) != len(features) or len(scaler["std"]) != len(features):
        raise ArtifactError(
            "scaler mean/std length does not match feature count."
        )

    if "lstm_ae" not in threshold_config:
        raise ArtifactError("threshold_config.json missing 'lstm_ae'.")
    threshold = float(threshold_config["lstm_ae"])
    if threshold <= 0 or not (threshold == threshold):  # NaN check
        raise ArtifactError(f"Invalid lstm_ae threshold: {threshold}")

    # PRD cited ~4.42494; notebook export used ~4.035 — accept whatever is saved
    if abs(threshold - 4.42494) > 1.0 and abs(threshold - 4.03496) > 1.0:
        warnings.append(
            f"Unusual threshold value {threshold:.5f}; verify it belongs to "
            "this checkpoint before trusting flagged rates."
        )

    thr_variant = threshold_config.get("variant")
    schema_variant = feature_schema.get("deployed_variant")
    if thr_variant and schema_variant and thr_variant != schema_variant:
        raise ArtifactError(
            f"Variant mismatch: threshold={thr_variant}, schema={schema_variant}."
        )

    win_samples = int(model_config.get("win_samples", 30))
    stride_samples = int(
        model_config.get("stride_samples")
        or feature_schema.get("window", {}).get("stride_samples")
        or 15
    )
    if win_samples != 30 or stride_samples != 15:
        warnings.append(
            f"Non-default window config: win={win_samples}, stride={stride_samples}."
        )

    return ModelArtifacts(
        model_dir=d,
        model_config=model_config,
        feature_schema=feature_schema,
        scaler=scaler,
        threshold_config=threshold_config,
        checkpoint_path=checkpoint_path,
        features=features,
        win_samples=win_samples,
        stride_samples=stride_samples,
        threshold=threshold,
        hidden_dim=int(model_config.get("hidden_dim", 48)),
        latent_dim=int(model_config.get("latent_dim", 12)),
        num_layers=int(model_config.get("num_layers", 1)),
        n_features=n_features,
        warnings=warnings,
    )
