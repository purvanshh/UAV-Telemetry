"""Reproducible threshold calibration (candidate artifacts only).

Nothing here writes ``uav_model/threshold_config.json``. A calibration run
produces a *candidate* artifact and a comparison with the deployed threshold;
promotion is a separate, manual decision.

Semantics, all documented in the artifact itself:

* Scores are the deployed scores: ``align_flight`` → ``make_windows`` (30/15) →
  standardise with ``scaler.json`` → mean squared reconstruction error
  (``UAVDetector.recon_error_per_window``, float32).
* Threshold = ``numpy.quantile(scores, q, method="linear")`` over *all* scored
  windows of the calibration flights pooled (a window-level, not flight-level,
  quantile).
* A window is flagged iff ``score > threshold`` (strict), as in
  ``UAVDetector.score``.
* Only flights whose manifest ``split == "val"`` **and** ``calibration_eligible``
  is true (a human declaration that the flight is representative behaviour) are
  used. Test flights are never opened by this module.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.artifact_loader import ModelArtifacts
from src.detector import UAVDetector
from src.flight_identity import STREAMS
from src.preprocessing_adapter import align_uploaded_flight_detailed
from src.provenance import artifact_fingerprints
from src.threshold_sensitivity import score_distribution
from src.validation import read_csv_bytes
from src.window_generator import generate_windows

QUANTILE_METHOD = "linear"
SCHEMA = "threshold_calibration/1"
MIN_CALIBRATION_WINDOWS = 200  # refuse a 0.995 quantile from fewer scores than this


class CalibrationError(RuntimeError):
    pass


def load_flight_frames(path: Path | str) -> Dict[str, pd.DataFrame]:
    """Load IMU/ATT/BARO from a directory of ``{IMU,ATT,BARO}.parquet`` or CSVs."""
    from src.upload import classify_stream_from_name

    d = Path(path)
    frames: Dict[str, pd.DataFrame] = {}
    for s in STREAMS:
        pq = d / f"{s}.parquet"
        if pq.is_file():
            frames[s] = pd.read_parquet(pq)
    if len(frames) == 3:
        return frames
    for f in sorted(d.glob("*.csv")):
        s = classify_stream_from_name(f.name)
        if s and s not in frames:
            frames[s] = read_csv_bytes(f.read_bytes(), filename=f.name)
    missing = [s for s in STREAMS if s not in frames]
    if missing:
        raise CalibrationError(f"{d}: missing stream(s) {missing}")
    return frames


def score_flight_frames(
    frames: Mapping[str, pd.DataFrame],
    artifacts: ModelArtifacts,
    detector: UAVDetector,
    flight_id: str = "",
) -> Dict[str, Any]:
    """Deployed scoring of one flight. Returns windows table + accounting."""
    detail = align_uploaded_flight_detailed(dict(frames), artifacts.model_dir)
    batch = generate_windows(detail.aligned, artifacts, model_dir=artifacts.model_dir)
    scores, flags, per_feat = detector.score(batch.X)
    rows = []
    for i, m in enumerate(batch.meta):
        row = {
            "flight_id": flight_id,
            "window_index": i,
            "session_id": m["session_id"],
            "t_start": m["t_start"],
            "t_end": m["t_end"],
            "anomaly_score": float(scores[i]),
            "threshold": float(artifacts.threshold),
            "is_anomaly": bool(flags[i]),
        }
        for j, f in enumerate(artifacts.features):
            row[f"feat_err_{f}"] = float(per_feat[i, j])
        rows.append(row)
    return {
        "windows": pd.DataFrame(rows),
        "scores32": np.asarray(scores, dtype=np.float32),
        "aligned": detail.aligned,
        "detail": detail,
    }


def quantile_threshold(scores: Sequence[float], q: float) -> float:
    arr = np.asarray(scores, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if not 0.0 < q < 1.0:
        raise CalibrationError(f"quantile must be in (0, 1); got {q}")
    if len(arr) == 0:
        raise CalibrationError("no finite scores")
    return float(np.quantile(arr, q, method=QUANTILE_METHOD))


def calibration_flights(manifest: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Flights eligible for calibration: split == val and calibration_eligible."""
    return [
        f
        for f in manifest.get("flights", [])
        if f.get("split") == "val"
        and f.get("calibration_eligible") is True
        and f.get("data_available", True)
    ]


def calibrate(
    manifest: Mapping[str, Any],
    *,
    artifacts: ModelArtifacts,
    detector: UAVDetector,
    resolve: Callable[[Mapping[str, Any]], Dict[str, pd.DataFrame]],
    quantile: Optional[float] = None,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a candidate calibration artifact (never promotes it).

    ``quantile`` defaults to the deployed config's recorded ``quantile_used``.
    Returns ``{"status": "no_validation_data", ...}`` rather than inventing data
    when no eligible validation flights exist.
    """
    q = float(quantile if quantile is not None else artifacts.threshold_config.get("quantile_used", 0.995))
    base = {
        "schema": SCHEMA,
        "status": "candidate",
        "promoted": False,
        "calibration_timestamp": timestamp or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "quantile": q,
        "quantile_method": f"numpy.quantile(float32 scores, q, method='{QUANTILE_METHOD}')",
        "comparison_operator": "score > threshold",
        "pooling": "all scored windows of all calibration flights pooled",
        "model_artifacts_sha256": artifact_fingerprints(artifacts),
        "preprocessing": {
            "features": list(artifacts.features),
            "win_samples": artifacts.win_samples,
            "stride_samples": artifacts.stride_samples,
            "alignment": {k: v for k, v in _align_cfg(artifacts).items()},
        },
        "manifest": {
            "kind": manifest.get("kind"),
            "created": manifest.get("created"),
            "n_flights_listed": len(manifest.get("flights", [])),
        },
        "deployed": {
            "threshold": artifacts.threshold,
            "quantile_recorded": artifacts.threshold_config.get("quantile_used"),
        },
    }
    eligible = calibration_flights(manifest)
    if not eligible:
        return {
            **base,
            "status": "no_validation_data",
            "reason": (
                "No manifest flight has split='val' AND calibration_eligible=true with data "
                "available. No threshold was computed; none was fabricated."
            ),
            "threshold": None,
        }

    all_scores: List[np.ndarray] = []
    per_flight = []
    for f in eligible:
        r = score_flight_frames(resolve(f), artifacts, detector, f["flight_id"])
        n = int(len(r["scores32"]))
        per_flight.append(
            {
                "flight_id": f["flight_id"],
                "n_windows": n,
                "content_hash": f.get("content_hash"),
                "flagged_at_deployed": int((r["scores32"] > np.float32(artifacts.threshold)).sum()),
            }
        )
        all_scores.append(r["scores32"])
    pooled = np.concatenate(all_scores) if all_scores else np.array([], dtype=np.float32)
    n_windows = int(len(pooled))
    if n_windows < MIN_CALIBRATION_WINDOWS:
        return {
            **base,
            "status": "insufficient_validation_data",
            "reason": f"{n_windows} validation windows < minimum {MIN_CALIBRATION_WINDOWS} "
            f"for a q={q} quantile.",
            "n_windows": n_windows,
            "flight_ids": [p["flight_id"] for p in per_flight],
            "threshold": None,
        }
    thr = quantile_threshold(pooled, q)
    dep = artifacts.threshold
    return {
        **base,
        "threshold": thr,
        "n_windows": n_windows,
        "n_flights": len(per_flight),
        "flight_ids": sorted(p["flight_id"] for p in per_flight),
        "per_flight": per_flight,
        "score_distribution": score_distribution(pooled),
        "comparison_with_deployed": {
            "deployed_threshold": dep,
            "candidate_threshold": thr,
            "abs_diff": abs(thr - dep),
            "relative_diff": (thr - dep) / dep,
            "reproduces_deployed": bool(abs(thr - dep) <= 1e-6 * max(1.0, abs(dep))),
            "validation_flagged_at_deployed": int((pooled > np.float32(dep)).sum()),
            "validation_flagged_at_candidate": int((pooled > np.float32(thr)).sum()),
            "note": "Reproduction is only meaningful if the manifest is the historical "
            "validation set with the original preprocessing; otherwise this is a new calibration.",
        },
    }


def _align_cfg(artifacts: ModelArtifacts) -> Dict[str, Any]:
    from src.preprocessing_adapter import _import_prep

    prep = _import_prep(artifacts.model_dir)
    return {k: prep.DEFAULT_ALIGN[k] for k in ("target_hz", "max_gap_s", "session_gap_s")}


def write_candidate(result: Mapping[str, Any], out_dir: Path | str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "threshold_candidate.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return path
