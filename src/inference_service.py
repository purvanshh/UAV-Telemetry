"""End-to-end inference orchestration for an uploaded flight."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.artifact_loader import ArtifactError, ModelArtifacts, load_artifacts
from src.detector import DetectorError, UAVDetector
from src.event_builder import (
    DEFAULT_MAX_GAP_S,
    AnomalyEvent,
    build_events,
    events_to_dataframe,
)
from src.preprocessing_adapter import (
    PreprocessingError,
    align_uploaded_flight,
    data_quality_warnings,
)
from src.report_builder import build_flight_report, events_dataframe_from_report
from src.report_schema import FlightReport
from src.window_generator import WindowBatch, generate_windows


@dataclass
class AnalysisResult:
    ok: bool
    status: str  # completed | insufficient_data | error | no_anomaly
    flight_id: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    n_windows: int = 0
    n_flagged: int = 0
    flagged_rate: Optional[float] = None
    n_events: int = 0
    threshold: float = 0.0
    variant: str = ""
    features: List[str] = field(default_factory=list)
    win_samples: int = 30
    stride_samples: int = 15
    feature_schema_version: Optional[str] = None
    score_mean: Optional[float] = None
    score_p50: Optional[float] = None
    score_p95: Optional[float] = None
    score_max: Optional[float] = None
    windows: Optional[pd.DataFrame] = None
    events: List[AnomalyEvent] = field(default_factory=list)
    events_df: Optional[pd.DataFrame] = None
    aligned: Optional[pd.DataFrame] = None
    report: Optional[FlightReport] = None
    artifact_warnings: List[str] = field(default_factory=list)
    max_gap_s: float = DEFAULT_MAX_GAP_S

    def summary_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("windows", None)
        d.pop("events", None)
        d.pop("events_df", None)
        d.pop("aligned", None)
        d.pop("report", None)
        return d


_detector_cache: Dict[str, UAVDetector] = {}
_artifacts_cache: Dict[str, ModelArtifacts] = {}


def get_artifacts(model_dir: Optional[str] = None) -> ModelArtifacts:
    key = str(model_dir or "default")
    if key not in _artifacts_cache:
        _artifacts_cache[key] = load_artifacts(model_dir)
    return _artifacts_cache[key]


def get_detector(model_dir: Optional[str] = None) -> UAVDetector:
    key = str(model_dir or "default")
    if key not in _detector_cache:
        arts = get_artifacts(model_dir)
        _detector_cache[key] = UAVDetector(arts)
    return _detector_cache[key]


def clear_caches() -> None:
    _detector_cache.clear()
    _artifacts_cache.clear()


def _empty_report(
    *,
    flight_id: str,
    status: str,
    artifacts: Optional[ModelArtifacts],
    warnings: List[str],
    errors: List[str],
    windows: Optional[pd.DataFrame] = None,
) -> Optional[FlightReport]:
    if artifacts is None:
        return None
    try:
        return build_flight_report(
            flight_id=flight_id,
            analysis_status=status,
            windows=windows,
            threshold=artifacts.threshold,
            variant=artifacts.variant,
            features=list(artifacts.features),
            win_samples=artifacts.win_samples,
            stride_samples=artifacts.stride_samples,
            feature_schema_version=artifacts.feature_schema.get("version"),
            data_quality_warnings=warnings,
            errors=errors,
        )
    except Exception:
        return None


def analyze_flight(
    frames: Dict[str, pd.DataFrame],
    flight_id: str = "",
    model_dir: Optional[str] = None,
    upload_warnings: Optional[List[str]] = None,
    max_gap_s: float = DEFAULT_MAX_GAP_S,
) -> AnalysisResult:
    """Run preprocessing → windows → LSTM scoring → event grouping → report."""
    warnings: List[str] = list(upload_warnings or [])
    try:
        artifacts = get_artifacts(model_dir)
    except ArtifactError as exc:
        return AnalysisResult(
            ok=False,
            status="error",
            flight_id=flight_id,
            errors=[str(exc)],
        )

    warnings.extend(artifacts.warnings)
    schema_ver = artifacts.feature_schema.get("version")

    try:
        aligned, _prep = align_uploaded_flight(frames, model_dir=model_dir)
    except PreprocessingError as exc:
        return AnalysisResult(
            ok=False,
            status="error",
            flight_id=flight_id,
            errors=[str(exc)],
            warnings=warnings,
            threshold=artifacts.threshold,
            variant=artifacts.variant,
            features=list(artifacts.features),
            win_samples=artifacts.win_samples,
            stride_samples=artifacts.stride_samples,
            feature_schema_version=schema_ver,
            report=_empty_report(
                flight_id=flight_id,
                status="error",
                artifacts=artifacts,
                warnings=warnings,
                errors=[str(exc)],
            ),
        )

    warnings.extend(data_quality_warnings(aligned))

    try:
        batch: WindowBatch = generate_windows(aligned, artifacts, model_dir=model_dir)
    except PreprocessingError as exc:
        return AnalysisResult(
            ok=False,
            status="error",
            flight_id=flight_id,
            errors=[str(exc)],
            warnings=warnings,
            threshold=artifacts.threshold,
            variant=artifacts.variant,
            features=list(artifacts.features),
            win_samples=artifacts.win_samples,
            stride_samples=artifacts.stride_samples,
            feature_schema_version=schema_ver,
            aligned=aligned,
            report=_empty_report(
                flight_id=flight_id,
                status="error",
                artifacts=artifacts,
                warnings=warnings,
                errors=[str(exc)],
            ),
        )

    if batch.n_windows == 0:
        errs = [
            "No valid windows could be generated from this flight. "
            "The flight may be too short, dominated by gaps, or lack "
            "finite values in the 12 deployed IMU/ATT features."
        ]
        return AnalysisResult(
            ok=False,
            status="insufficient_data",
            flight_id=flight_id,
            errors=errs,
            warnings=warnings,
            threshold=artifacts.threshold,
            variant=artifacts.variant,
            features=list(artifacts.features),
            win_samples=artifacts.win_samples,
            stride_samples=artifacts.stride_samples,
            feature_schema_version=schema_ver,
            n_windows=0,
            flagged_rate=None,
            aligned=aligned,
            artifact_warnings=list(artifacts.warnings),
            report=_empty_report(
                flight_id=flight_id,
                status="insufficient_data",
                artifacts=artifacts,
                warnings=warnings,
                errors=errs,
            ),
        )

    try:
        detector = get_detector(model_dir)
        scores, flags, per_feat = detector.score(batch.X)
    except (DetectorError, Exception) as exc:
        return AnalysisResult(
            ok=False,
            status="error",
            flight_id=flight_id,
            errors=[f"Inference failed: {exc}"],
            warnings=warnings,
            threshold=artifacts.threshold,
            variant=artifacts.variant,
            features=list(artifacts.features),
            win_samples=artifacts.win_samples,
            stride_samples=artifacts.stride_samples,
            feature_schema_version=schema_ver,
            n_windows=batch.n_windows,
            aligned=aligned,
        )

    rows = []
    for i, m in enumerate(batch.meta):
        row = {
            "flight_id": flight_id,
            "window_index": i,
            "session_id": m.get("session_id"),
            "t_start": m["t_start"],
            "t_end": m["t_end"],
            "anomaly_score": float(scores[i]),
            "threshold": float(artifacts.threshold),
            "is_anomaly": bool(flags[i]),
        }
        for j, feat in enumerate(artifacts.features):
            row[f"feat_err_{feat}"] = float(per_feat[i, j])
        rows.append(row)
    windows_df = pd.DataFrame(rows)

    n_flagged = int(flags.sum())
    events = build_events(
        windows_df, max_gap_s=max_gap_s, features=list(artifacts.features)
    )
    events_df = events_to_dataframe(events, float(artifacts.threshold))

    status = "completed"
    if n_flagged == 0:
        status = "no_anomaly"

    report = build_flight_report(
        flight_id=flight_id,
        analysis_status=status,
        windows=windows_df,
        threshold=float(artifacts.threshold),
        variant=artifacts.variant,
        features=list(artifacts.features),
        win_samples=artifacts.win_samples,
        stride_samples=artifacts.stride_samples,
        feature_schema_version=schema_ver,
        data_quality_warnings=warnings,
        errors=[],
        aligned=aligned,
        max_gap_s=max_gap_s,
        score_mean=float(np.mean(scores)),
        score_p50=float(np.quantile(scores, 0.50)),
        score_p95=float(np.quantile(scores, 0.95)),
        score_max=float(np.max(scores)),
    )
    # Prefer report-derived event CSV (includes affected_signals)
    events_df = events_dataframe_from_report(report)

    result = AnalysisResult(
        ok=True,
        status=status,
        flight_id=flight_id,
        warnings=warnings,
        n_windows=int(len(scores)),
        n_flagged=n_flagged,
        flagged_rate=float(n_flagged / len(scores)) if len(scores) else None,
        n_events=len(events),
        threshold=float(artifacts.threshold),
        variant=artifacts.variant,
        features=list(artifacts.features),
        win_samples=artifacts.win_samples,
        stride_samples=artifacts.stride_samples,
        feature_schema_version=schema_ver,
        score_mean=float(np.mean(scores)),
        score_p50=float(np.quantile(scores, 0.50)),
        score_p95=float(np.quantile(scores, 0.95)),
        score_max=float(np.max(scores)),
        windows=windows_df,
        events=events,
        events_df=events_df,
        aligned=aligned,
        report=report,
        artifact_warnings=list(artifacts.warnings),
        max_gap_s=max_gap_s,
    )

    if n_flagged == 0:
        result.warnings.append(
            "No windows exceeded the saved threshold. This is not a guarantee "
            "that the flight is free of anomalies — the detector only scores "
            "the 12 conservative IMU/ATT features."
        )
    return result
