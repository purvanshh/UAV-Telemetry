"""Adapter around ``uav_model/preprocessing.py`` for in-memory uploads."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.artifact_loader import resolve_model_dir


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


def _ensure_stream_df(prep, df: pd.DataFrame, spec: Dict[str, Any]) -> pd.DataFrame:
    if hasattr(prep, "load_stream_df"):
        return prep.load_stream_df(df, spec)
    # Fallback for notebook exports that only expose load_stream(path, spec)
    unit_mult = getattr(prep, "UNIT_MULT", {"us": 1e6})
    resolve = prep.resolve
    ts_col = None
    for c in spec["ts_priority"]:
        r = resolve(list(df.columns), c)
        if r is not None:
            ts_col = r
            break
    if ts_col is None:
        raise ValueError("no timestamp column")
    t_raw = pd.to_numeric(df[ts_col], errors="coerce").astype("float64")
    unit = spec.get("ts_unit", "us")
    out = df.copy()
    out["__t_s"] = (t_raw / unit_mult[unit]).values
    out = (
        out.dropna(subset=["__t_s"])
        .sort_values("__t_s")
        .drop_duplicates("__t_s")
        .reset_index(drop=True)
    )
    return out


def align_uploaded_flight(
    frames: Dict[str, pd.DataFrame],
    model_dir: Optional[Path | str] = None,
    align_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Any]:
    """Load IMU/ATT/BARO DataFrames into the notebook preprocessing pipeline.

    Returns ``(aligned_df, prep_module)``.
    """
    for stream in ("IMU", "ATT", "BARO"):
        if stream not in frames or frames[stream] is None:
            raise PreprocessingError(f"Missing {stream} frame for alignment.")

    prep = _import_prep(model_dir)
    schema = prep.DEFAULT_SCHEMA
    cfg = dict(prep.DEFAULT_ALIGN)
    if align_cfg:
        cfg.update(align_cfg)

    try:
        imu = _ensure_stream_df(prep, frames["IMU"], schema["IMU"])
        att = _ensure_stream_df(prep, frames["ATT"], schema["ATT"])
        baro = _ensure_stream_df(prep, frames["BARO"], schema["BARO"])
        aligned = prep.align_flight(imu, att, baro, schema, cfg)
    except Exception as exc:
        raise PreprocessingError(f"Preprocessing failed: {exc}") from exc

    if aligned.empty:
        raise PreprocessingError("Aligned flight is empty — no temporal overlap.")
    if "_session_id" not in aligned.columns:
        raise PreprocessingError(
            "Aligned output is missing _session_id; check preprocessing.py version."
        )
    return aligned, prep


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
