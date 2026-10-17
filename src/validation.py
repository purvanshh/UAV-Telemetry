"""Validate IMU / ATT / BARO telemetry CSVs against the notebook schema.

Validation is non-destructive: extreme values and quality issues are reported
as warnings. Hard errors (missing files, unreadable CSV, missing required
columns, unusable timestamps) reject the stream.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.schema import (
    STREAM_SCHEMA,
    STREAMS,
)
from src.timebase import SUPPORTED_TIMESTAMP_FIELDS, normalize_timestamps


def _norm(s: str) -> str:
    return re.sub(r"[\s_]+", "", str(s)).lower()


def resolve_column(columns: List[str], name: str) -> Optional[str]:
    """Resolve a logical column name against actual CSV headers (case/space tolerant)."""
    lut = {_norm(c): c for c in columns}
    return lut.get(_norm(name))


@dataclass
class StreamValidationResult:
    stream: str
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    filename: Optional[str] = None
    file_size_bytes: Optional[int] = None
    n_rows: int = 0
    n_cols: int = 0
    columns: List[str] = field(default_factory=list)
    timestamp_column: Optional[str] = None
    timestamp_unit: Optional[str] = None
    timestamp_unit_source: Optional[str] = None
    n_valid_timestamps: int = 0
    n_missing_timestamps: int = 0
    gaps: List[Dict[str, float]] = field(default_factory=list)
    t_start: Optional[float] = None
    t_end: Optional[float] = None
    duration_s: Optional[float] = None
    median_dt_s: Optional[float] = None
    est_hz: Optional[float] = None
    missing_value_frac: float = 0.0
    duplicate_timestamps: int = 0
    non_monotonic_timestamps: int = 0
    range_flags: List[str] = field(default_factory=list)
    sample_rows: List[Dict[str, Any]] = field(default_factory=list)
    resolved_features: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FlightValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    streams: Dict[str, StreamValidationResult] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "streams": {k: v.to_dict() for k, v in self.streams.items()},
        }


def read_csv_bytes(
    data: Union[bytes, bytearray, io.BytesIO, io.BufferedIOBase],
    filename: str = "",
) -> pd.DataFrame:
    """Parse CSV bytes into a DataFrame, raising ValueError on failure."""
    try:
        if isinstance(data, (bytes, bytearray)):
            buf = io.BytesIO(data)
        else:
            buf = data
            if hasattr(buf, "seek"):
                buf.seek(0)
        df = pd.read_csv(buf)
    except Exception as exc:
        raise ValueError(
            f"Could not parse '{filename or 'file'}' as CSV: {exc}"
        ) from exc
    if df is None or df.empty:
        raise ValueError(
            f"'{filename or 'file'}' is empty or has no data rows."
        )
    return df


def validate_stream(
    df: pd.DataFrame,
    stream: str,
    *,
    filename: Optional[str] = None,
    file_size_bytes: Optional[int] = None,
) -> StreamValidationResult:
    """Validate a single sensor stream DataFrame."""
    stream = stream.upper()
    result = StreamValidationResult(
        stream=stream,
        ok=True,
        filename=filename,
        file_size_bytes=file_size_bytes,
        n_rows=int(len(df)),
        n_cols=int(df.shape[1]),
        columns=[str(c) for c in df.columns],
    )

    if stream not in STREAM_SCHEMA:
        result.ok = False
        result.errors.append(f"Unknown stream type '{stream}'.")
        return result

    if len(df) == 0:
        result.ok = False
        result.errors.append(f"{stream}: file has no data rows.")
        return result

    cols = list(df.columns)
    spec = STREAM_SCHEMA[stream]

    # Required feature columns
    missing_required = []
    for feat, src in spec["required_features"].items():
        resolved = resolve_column(cols, src)
        if resolved is None:
            missing_required.append(src)
        else:
            result.resolved_features[feat] = resolved

    if missing_required:
        result.ok = False
        result.errors.append(
            f"{stream}: missing required column(s): {', '.join(missing_required)}. "
            f"Expected sensor columns matching the UAV dataset schema."
        )

    for feat, src in spec.get("optional_features", {}).items():
        resolved = resolve_column(cols, src)
        if resolved is not None:
            result.resolved_features[feat] = resolved

    # Timestamp — same interpretation as preprocessing (src/timebase.py)
    _clean, tb = normalize_timestamps(df, stream=stream)
    result.timestamp_column = tb.column
    result.timestamp_unit = tb.unit
    result.timestamp_unit_source = tb.unit_source
    result.n_valid_timestamps = tb.n_valid
    result.n_missing_timestamps = tb.n_missing
    result.duplicate_timestamps = tb.n_duplicates
    result.non_monotonic_timestamps = tb.n_nonmonotonic_steps
    result.gaps = list(tb.gaps)
    result.warnings.extend(tb.warnings)
    if tb.errors:
        result.ok = False
        result.errors.extend(tb.errors)
    else:
        result.t_start = tb.t_start_s
        result.t_end = tb.t_end_s
        result.duration_s = tb.duration_s
        result.median_dt_s = tb.median_dt_s
        result.est_hz = tb.est_hz

    # Missing values
    result.missing_value_frac = float(df.isna().mean().mean()) if df.shape[1] else 0.0
    if result.missing_value_frac > 0.05:
        result.warnings.append(
            f"{stream}: overall missing-value fraction is "
            f"{result.missing_value_frac:.1%}."
        )

    # Range checks — warn only, never reject
    for feat, (lo, hi) in spec.get("range_checks", {}).items():
        src = result.resolved_features.get(feat)
        if src is None:
            continue
        v = pd.to_numeric(df[src], errors="coerce").to_numpy(dtype="float64")
        v = v[np.isfinite(v)]
        if not len(v):
            continue
        if v.min() < lo:
            msg = f"{feat}: min {v.min():.3g} < {lo}"
            result.range_flags.append(msg)
            result.warnings.append(f"{stream}: suspicious range — {msg}")
        if v.max() > hi:
            msg = f"{feat}: max {v.max():.3g} > {hi}"
            result.range_flags.append(msg)
            result.warnings.append(f"{stream}: suspicious range — {msg}")

    # Sample preview (first 5 rows, JSON-safe)
    try:
        sample = df.head(5).copy()
        for c in sample.columns:
            if pd.api.types.is_datetime64_any_dtype(sample[c]):
                sample[c] = sample[c].astype(str)
        result.sample_rows = sample.where(pd.notnull(sample), None).to_dict(
            orient="records"
        )
    except Exception:
        result.sample_rows = []

    return result


def validate_flight(
    frames: Dict[str, pd.DataFrame],
    *,
    filenames: Optional[Dict[str, str]] = None,
    file_sizes: Optional[Dict[str, int]] = None,
) -> FlightValidationResult:
    """Validate that all three streams are present and individually valid.

    Different row counts / sampling rates across streams are allowed.
    """
    filenames = filenames or {}
    file_sizes = file_sizes or {}
    flight = FlightValidationResult(ok=True)

    missing = [s for s in STREAMS if s not in frames or frames[s] is None]
    if missing:
        flight.ok = False
        flight.errors.append(
            f"Missing required stream(s): {', '.join(missing)}. "
            f"Upload IMU, ATT and BARO CSVs (or a ZIP containing all three)."
        )

    for stream in STREAMS:
        if stream not in frames or frames[stream] is None:
            continue
        sv = validate_stream(
            frames[stream],
            stream,
            filename=filenames.get(stream),
            file_size_bytes=file_sizes.get(stream),
        )
        flight.streams[stream] = sv
        if not sv.ok:
            flight.ok = False
            flight.errors.extend(sv.errors)
        flight.warnings.extend(sv.warnings)

    # Cross-stream informational note (not an error)
    present = [s for s in STREAMS if s in flight.streams]
    if len(present) >= 2:
        counts = {s: flight.streams[s].n_rows for s in present}
        if len(set(counts.values())) > 1:
            flight.warnings.append(
                "Streams have different row counts "
                f"({', '.join(f'{k}={v}' for k, v in counts.items())}), "
                "which is expected for different sampling rates."
            )

    return flight
