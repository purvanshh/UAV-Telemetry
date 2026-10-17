"""Adapter around ``uav_model/preprocessing.py`` for in-memory uploads."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.artifact_loader import resolve_model_dir
from src.timebase import TimebaseReport, normalize_timestamps


class PreprocessingError(RuntimeError):
    """Raised when alignment / stream loading fails."""


def _import_prep(model_dir: Optional[Path | str] = None):
    """Load preprocessing.py from the given model dir by file path.

    Avoids stale ``sys.modules['preprocessing']`` collisions across fixture
    and production model directories.
    """
    d = resolve_model_dir(model_dir)
    prep_path = d / "preprocessing.py"
    if not prep_path.exists():
        raise PreprocessingError(f"preprocessing.py not found in {d}")

    module_name = f"uav_preprocessing_{abs(hash(str(prep_path.resolve())))}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, prep_path)
    if spec is None or spec.loader is None:
        raise PreprocessingError(f"Could not load {prep_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


@dataclass
class AlignmentDetail:
    """Aligned frame plus everything needed to audit how it was produced."""

    aligned: pd.DataFrame
    prep: Any
    cfg: Dict[str, Any]
    timebase: Dict[str, TimebaseReport]  # per stream, raw-input accounting
    clean_streams: Dict[str, pd.DataFrame]  # sorted/de-duplicated, with __t_s


def _normalize_stream(df: pd.DataFrame, stream: str, max_gap_s: float):
    """Timestamp handling shared with upload validation (src/timebase.py)."""
    clean, report = normalize_timestamps(df, stream=stream, gap_threshold_s=max_gap_s)
    if report.errors:
        raise PreprocessingError("; ".join(report.errors))
    return clean, report


def align_uploaded_flight_detailed(
    frames: Dict[str, pd.DataFrame],
    model_dir: Optional[Path | str] = None,
    align_cfg: Optional[Dict[str, Any]] = None,
) -> AlignmentDetail:
    for stream in ("IMU", "ATT", "BARO"):
        if stream not in frames or frames[stream] is None:
            raise PreprocessingError(f"Missing {stream} frame for alignment.")

    prep = _import_prep(model_dir)
    schema = prep.DEFAULT_SCHEMA
    cfg = dict(prep.DEFAULT_ALIGN)
    if align_cfg:
        cfg.update(align_cfg)

    try:
        clean: Dict[str, pd.DataFrame] = {}
        reports: Dict[str, TimebaseReport] = {}
        for stream in ("IMU", "ATT", "BARO"):
            clean[stream], reports[stream] = _normalize_stream(
                frames[stream], stream, float(cfg["max_gap_s"])
            )
        aligned = prep.align_flight(
            clean["IMU"], clean["ATT"], clean["BARO"], schema, cfg
        )
    except PreprocessingError:
        raise
    except Exception as exc:
        raise PreprocessingError(f"Preprocessing failed: {exc}") from exc

    if aligned.empty:
        raise PreprocessingError("Aligned flight is empty — no temporal overlap.")
    if "_session_id" not in aligned.columns:
        raise PreprocessingError(
            "Aligned output is missing _session_id; check preprocessing.py version."
        )
    return AlignmentDetail(
        aligned=aligned, prep=prep, cfg=cfg, timebase=reports, clean_streams=clean
    )


def align_uploaded_flight(
    frames: Dict[str, pd.DataFrame],
    model_dir: Optional[Path | str] = None,
    align_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Any]:
    """Load IMU/ATT/BARO DataFrames into the notebook preprocessing pipeline.

    Returns ``(aligned_df, prep_module)``.
    """
    d = align_uploaded_flight_detailed(frames, model_dir, align_cfg)
    return d.aligned, d.prep


def data_quality_warnings(aligned: pd.DataFrame) -> list[str]:
    """Surface BARO extremes without implying LSTM coverage of those signals."""
    warnings: list[str] = []
    if "alt" in aligned.columns and (aligned["alt"] < -1000).any():
        warnings.append(
            "BARO altitude contains unresolved extreme values. "
            "The deployed LSTM excludes altitude — this does not produce "
            "altitude anomaly detections."
        )
    if "press" in aligned.columns and (aligned["press"] > 1.2e5).any():
        warnings.append(
            "BARO pressure exceeds typical surface range. "
            "The deployed LSTM excludes pressure — flagged windows reflect "
            "IMU/ATT reconstruction only."
        )
    sids = aligned["_session_id"].values
    n_sessions = len({int(s) for s in sids if s >= 0})
    if n_sessions > 1:
        warnings.append(
            f"Flight contains {n_sessions} sessions separated by source gaps; "
            "windows never cross session boundaries."
        )
    return warnings
