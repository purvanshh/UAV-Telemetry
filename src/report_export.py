"""Serialize validated reports and CSV tables for download."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from src.report_schema import FlightReport, validate_report


def report_to_json_bytes(report: FlightReport, *, indent: int = 2) -> bytes:
    """Validate then serialize. Raises ValidationError if invalid."""
    validated = validate_report(report)
    text = json.dumps(validated.to_json_dict(), indent=indent, allow_nan=False)
    return text.encode("utf-8")


def windows_to_csv_bytes(windows: Optional[pd.DataFrame]) -> bytes:
    if windows is None or len(windows) == 0:
        return b""
    return windows.to_csv(index=False).encode("utf-8")


def events_to_csv_bytes(events_df: Optional[pd.DataFrame]) -> bytes:
    if events_df is None or len(events_df) == 0:
        # Header-only empty export for consistent downloads
        cols = [
            "event_id",
            "session_id",
            "start_time_s",
            "end_time_s",
            "duration_s",
            "flagged_window_count",
            "peak_score",
            "mean_flagged_score",
            "threshold",
            "peak_over_threshold",
            "threshold_relative_score",
            "affected_signals",
            "window_indices",
        ]
        return (",".join(cols) + "\n").encode("utf-8")
    return events_df.to_csv(index=False).encode("utf-8")


def write_report_json(
    report: FlightReport,
    path: Union[str, Path],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(report_to_json_bytes(report))
    return path
