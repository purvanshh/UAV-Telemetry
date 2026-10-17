"""Feature lineage, artifact fingerprints and run provenance for reports.

Units and sources come from ``uav_model/feature_schema.json`` (the schema the
model was built with). They describe what the *schema declares*; nothing here
inspects a log to verify units, and anything the schema does not state is
reported as ``"unknown"`` rather than invented.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.artifact_loader import ModelArtifacts
from src.schema import STREAM_SCHEMA

# Candidate features in preprocessing order, with the source of each.
# (stream, [source fields], transformation) — derived entries mirror
# uav_model/preprocessing.py::align_flight.
_DERIVED_SOURCES = {
    "yaw_sin": ("ATT", ["Yaw"], "sin(deg2rad(Yaw)) after resampling"),
    "yaw_cos": ("ATT", ["Yaw"], "cos(deg2rad(Yaw)) after resampling"),
    "roll_res": ("ATT", ["Roll", "DesRoll"], "Roll - DesRoll after resampling"),
    "pitch_res": ("ATT", ["Pitch", "DesPitch"], "Pitch - DesPitch after resampling"),
    "alt_rate": ("BARO", ["Alt"], "finite difference of resampled Alt per grid step, NaN across gaps"),
    "press_rate": ("BARO", ["Press"], "finite difference of resampled Press per grid step, NaN across gaps"),
}


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_fingerprints(artifacts: ModelArtifacts) -> Dict[str, str]:
    """SHA-256 of each deployed artifact file (checkpoint, scaler, configs, preprocessing)."""
    d = artifacts.model_dir
    out = {"checkpoint": sha256_file(artifacts.checkpoint_path)}
    for key, name in (
        ("scaler", "scaler.json"),
        ("threshold_config", "threshold_config.json"),
        ("model_config", "model_config.json"),
        ("feature_schema", "feature_schema.json"),
        ("preprocessing_py", "preprocessing.py"),
    ):
        p = d / name
        if p.is_file():
            out[key] = sha256_file(p)
    return out


def feature_provenance(
    artifacts: ModelArtifacts, candidate_features: List[str]
) -> List[Dict[str, Any]]:
    """One row per candidate feature: source, unit, transformation, role, model use."""
    schema = artifacts.feature_schema
    streams = schema.get("streams", {})
    derived = schema.get("derived_features", {})
    deployed = list(artifacts.features)
    scaler = artifacts.scaler
    rows: List[Dict[str, Any]] = []
    for feat in candidate_features:
        row: Dict[str, Any] = {"feature": feat}
        if feat in _DERIVED_SOURCES:
            stream, src, how = _DERIVED_SOURCES[feat]
            row.update(
                kind="derived",
                stream=stream,
                source_fields=src,
                unit=(derived.get(feat) or {}).get("units", "unknown"),
                transformation=how,
            )
        else:
            row.update(kind="raw", stream=None, source_fields=[], unit="unknown",
                       transformation="gap-aware linear interpolation to 10 Hz grid")
            for sname, sdef in streams.items():
                if feat in sdef.get("features", {}):
                    row.update(
                        stream=sname,
                        source_fields=[sdef["features"][feat]],
                        unit=(sdef.get("units") or {}).get(feat, "unknown"),
                    )
            if row["stream"] is None:  # schema file lacks a streams block: repo schema
                for sname, sdef in STREAM_SCHEMA.items():
                    feats = {**sdef["required_features"], **sdef.get("optional_features", {})}
                    if feat in feats:
                        row.update(
                            stream=sname,
                            source_fields=[feats[feat]],
                            unit=sdef["units"].get(feat, "unknown"),
                        )
        if feat in ("alt_rate", "press_rate"):
            # preprocessing.safe_diff does not divide by dt: value is change per 0.1 s grid step
            row["unit"] = f"{'m' if feat == 'alt_rate' else 'Pa'} per grid step (0.1 s)"
            row["unit_note"] = ("feature_schema.json declares per-second units, but preprocessing "
                                "differences without dividing by dt; not scored by the model")
        row["resampled"] = True  # every feature passes through the 10 Hz grid
        row["unit_basis"] = "declared by feature_schema.json; not verified from the uploaded log"
        used = feat in deployed
        row["used_by_model"] = used
        row["standardized"] = used
        if used:
            i = deployed.index(feat)
            row["model_input_index"] = i
            row["scaler_mean"] = float(scaler["mean"][i])
            row["scaler_std"] = float(scaler["std"][i])
            row["scaling"] = "(x - mean) / std using scaler.json (fit on training flights)"
        else:
            row["exclusion_reason"] = (
                "not in the deployed 'conservative' 12-feature set "
                f"(feature_schema.deployed_variant={schema.get('deployed_variant')!r})"
            )
        rows.append(row)
    return rows


def deployed_config(artifacts: ModelArtifacts, align_cfg: Dict[str, Any]) -> Dict[str, Any]:
    tc = artifacts.threshold_config
    return {
        "model": "LSTMAE",
        "variant": artifacts.variant,
        "features": list(artifacts.features),
        "input_shape": [artifacts.win_samples, len(artifacts.features)],
        "hidden_dim": artifacts.hidden_dim,
        "latent_dim": artifacts.latent_dim,
        "num_layers": artifacts.num_layers,
        "window": {
            "win_samples": artifacts.win_samples,
            "stride_samples": artifacts.stride_samples,
            "win_seconds": artifacts.win_samples / float(align_cfg["target_hz"]),
            "stride_seconds": artifacts.stride_samples / float(align_cfg["target_hz"]),
        },
        "alignment": {
            k: align_cfg[k] for k in ("target_hz", "max_gap_s", "session_gap_s") if k in align_cfg
        },
        "scaler": {
            "type": "per-feature z-score; std < 1e-12 replaced by 1.0",
            "fit_on_flights": len(artifacts.scaler.get("fit_on_flights") or []) or None,
        },
        "threshold": {
            "value": artifacts.threshold,
            "quantile_recorded": tc.get("quantile_used"),
            "calibrated_on_recorded": tc.get("calibrated_on"),
            "flag_rule": "anomaly_score > threshold (strict)",
            "score": "mean squared error over 30 samples x 12 standardized features",
        },
    }


def build_provenance(
    *,
    artifacts: ModelArtifacts,
    align_cfg: Dict[str, Any],
    candidate_features: List[str],
    filenames: Optional[Dict[str, str]] = None,
    file_hashes: Optional[Dict[str, str]] = None,
    flight_meta: Optional[Dict[str, Any]] = None,
    timebase: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    meta = flight_meta or {}
    return {
        "source": {
            "filenames": dict(filenames or {}),
            "file_sha256": dict(file_hashes or {}),
            "upload_source": meta.get("source"),
            "uploaded_at": meta.get("created_at"),
            "duplicate_of": meta.get("duplicate_of"),
        },
        "timestamps": {
            s: {
                "column": r.column,
                "unit": r.unit,
                "unit_source": r.unit_source,
                "conversion": "seconds = raw / 1e6" if r.unit == "us" else None,
            }
            for s, r in (timebase or {}).items()
        },
        "artifacts": artifact_fingerprints(artifacts),
        "deployed": deployed_config(artifacts, align_cfg),
        "features": feature_provenance(artifacts, candidate_features),
    }
