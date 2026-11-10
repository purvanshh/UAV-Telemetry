"""Build a concise, operator-readable upload validation summary for Deep Analysis.

This module produces a structured summary of what was found in each uploaded
file, including timestamp quality, data coverage, missing/incompatible columns,
and how many rows are usable.  It is a *read-only projection* of the data that
was already validated by ``src.validation``; it never modifies stored data.

The summary is returned from ``GET /api/flights/{id}/upload-summary`` and
displayed in the upload UI before the operator clicks "Run analysis".
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.schema import STREAM_SCHEMA, STREAMS
from src.validation import FlightValidationResult, StreamValidationResult


# ── human-readable signal descriptions for Deep Analysis ─────────────────────────

_STREAM_ROLE: Dict[str, str] = {
    "IMU": "Gyroscope and accelerometer (scored by the detector)",
    "ATT": "Roll, pitch, and yaw angles (scored by the detector)",
    "BARO": "Barometer — altitude and pressure (context only, not scored)",
}

_STREAM_REQUIRED: Dict[str, List[str]] = {
    "IMU": ["GyrX", "GyrY", "GyrZ", "AccX", "AccY", "AccZ"],
    "ATT": ["Roll", "Pitch", "Yaw", "DesRoll", "DesPitch"],
    "BARO": ["Alt", "Press"],
}

_COVERAGE_THRESHOLDS = {
    "ok":        (0.80, "Good coverage"),
    "low":       (0.50, "Low coverage — part of the flight cannot be assessed"),
    "very_low":  (0.0,  "Very low coverage — most of the flight cannot be assessed"),
}


def _coverage_label(fraction: Optional[float]) -> str:
    if fraction is None:
        return "Unknown"
    for key, (thresh, label) in _COVERAGE_THRESHOLDS.items():
        if fraction >= thresh:
            return label
    return "Very low coverage"


def _stream_summary(sv: StreamValidationResult) -> Dict[str, Any]:
    """Compact per-stream fact card."""
    out: Dict[str, Any] = {
        "stream": sv.stream,
        "role": _STREAM_ROLE.get(sv.stream, sv.stream),
        "ok": sv.ok,
        "filename": sv.filename,
        "file_size_bytes": sv.file_size_bytes,
        "rows": sv.n_rows,
        "columns": sv.n_cols,
        "errors": list(sv.errors),
        "warnings": list(sv.warnings)[:6],   # cap verbose warnings for the UI
        "timestamp": None,
        "missing_required": [],
        "range_flags": list(sv.range_flags or [])[:4],
    }

    # Timestamp quality
    if sv.timestamp_column:
        out["timestamp"] = {
            "column": sv.timestamp_column,
            "unit": sv.timestamp_unit,
            "unit_source": sv.timestamp_unit_source,
            "est_hz": round(sv.est_hz, 1) if sv.est_hz is not None else None,
            "duration_s": round(sv.duration_s, 1) if sv.duration_s is not None else None,
            "duplicate_timestamps": sv.duplicate_timestamps,
            "non_monotonic_timestamps": sv.non_monotonic_timestamps,
            "missing_timestamps": sv.n_missing_timestamps,
        }

    # Required columns missing or unresolvable
    req = _STREAM_REQUIRED.get(sv.stream, [])
    resolved_src = set(sv.resolved_features.values()) if sv.resolved_features else set()
    for col in req:
        # Check that the source column name (case-insensitive) is in resolved
        found = any(col.lower() in rk.lower() or rk.lower() in col.lower()
                    for rk in sv.resolved_features.values())
        if not found:
            out["missing_required"].append(col)

    return out


def build_upload_summary(
    meta: Dict[str, Any],
    validation: Optional[Dict[str, Any]],
    frames: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, Any]:
    """Build a UI-ready upload summary from stored meta and validation data.

    Parameters
    ----------
    meta:
        The flight's ``meta.json`` dict (as returned by ``FlightStore.load_meta``).
    validation:
        The ``validation`` section of ``meta.json`` (may be None for old flights).
    frames:
        Optional dict of DataFrames for richer per-stream stats.  When None the
        summary still works but per-stream row counts are taken from meta.
    """
    filenames = dict(meta.get("filenames") or {})
    file_sizes = dict(meta.get("file_sizes") or {})
    upload_warnings = list(meta.get("warnings") or [])
    source = meta.get("source", "unknown")

    # Per-stream validation from stored validation dict
    streams: List[Dict[str, Any]] = []
    val_streams: Dict[str, Any] = (validation or {}).get("streams") or {}

    for stream in STREAMS:
        if stream not in filenames and stream not in val_streams:
            continue
        sv_raw = val_streams.get(stream)
        if sv_raw and isinstance(sv_raw, dict):
            # Reconstruct a StreamValidationResult-like object from stored dict
            sv = StreamValidationResult(
                stream=stream,
                ok=bool(sv_raw.get("ok", True)),
                errors=list(sv_raw.get("errors") or []),
                warnings=list(sv_raw.get("warnings") or []),
                filename=sv_raw.get("filename") or filenames.get(stream),
                file_size_bytes=sv_raw.get("file_size_bytes")
                    or file_sizes.get(stream),
                n_rows=int(sv_raw.get("n_rows") or 0),
                n_cols=int(sv_raw.get("n_cols") or 0),
                columns=list(sv_raw.get("columns") or []),
                timestamp_column=sv_raw.get("timestamp_column"),
                timestamp_unit=sv_raw.get("timestamp_unit"),
                timestamp_unit_source=sv_raw.get("timestamp_unit_source"),
                est_hz=sv_raw.get("est_hz"),
                duration_s=sv_raw.get("duration_s"),
                t_start=sv_raw.get("t_start"),
                t_end=sv_raw.get("t_end"),
                duplicate_timestamps=int(sv_raw.get("duplicate_timestamps") or 0),
                non_monotonic_timestamps=int(sv_raw.get("non_monotonic_timestamps") or 0),
                n_missing_timestamps=int(sv_raw.get("n_missing_timestamps") or 0),
                missing_value_frac=float(sv_raw.get("missing_value_frac") or 0.0),
                range_flags=list(sv_raw.get("range_flags") or []),
                resolved_features=dict(sv_raw.get("resolved_features") or {}),
                n_valid_timestamps=int(sv_raw.get("n_valid_timestamps") or 0),
            )
        else:
            # Minimal entry — file was stored but no per-stream validation available
            sv = StreamValidationResult(
                stream=stream,
                ok=True,
                filename=filenames.get(stream),
                file_size_bytes=file_sizes.get(stream),
            )
            if frames and stream in frames:
                sv.n_rows = len(frames[stream])
                sv.n_cols = frames[stream].shape[1]
                sv.columns = list(frames[stream].columns)

        streams.append(_stream_summary(sv))

    # Overall readiness for Deep Analysis
    streams_ok = all(s["ok"] for s in streams)
    all_errors: List[str] = []
    for s in streams:
        all_errors.extend(s["errors"])

    # Quickly estimate overlap/duration for usability hint
    hints: List[str] = []
    if not streams_ok:
        hints.append(
            "One or more required files have errors. Correct them before running analysis."
        )
    elif upload_warnings:
        hints.append(
            "The files were accepted with warnings. Review them before running analysis."
        )
    else:
        hints.append(
            "All files passed validation. You can run the analysis."
        )

    return {
        "flight_id": meta.get("flight_id", "unknown"),
        "source": source,
        "filenames": filenames,
        "file_sizes": file_sizes,
        "upload_warnings": upload_warnings,
        "streams": streams,
        "all_errors": all_errors,
        "ready_for_analysis": streams_ok and not all_errors,
        "hints": hints,
        "analysis_note": (
            "Deep Analysis scores 12 gyroscope, accelerometer, and attitude signals. "
            "Barometer data provides context but is not scored by the anomaly detector."
        ),
    }
