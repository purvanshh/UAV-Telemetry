"""Resource limits for uploads and ZIP archives, and the flight-ID rule.

Limits are enforced before data is parsed or decompressed. ``UAV_MAX_UPLOAD_MB``
overrides the per-file cap (default 100 MB).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

DEFAULT_MAX_UPLOAD_MB = 100
MAX_ZIP_MEMBERS = 200
MAX_ZIP_RATIO = 100.0  # uncompressed / compressed, per member
MAX_CSV_ROWS = 5_000_000
MAX_CSV_COLUMNS = 200

FLIGHT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class LimitError(ValueError):
    """An upload exceeded a documented resource limit."""


class InvalidFlightId(ValueError):
    """The flight identifier is not a safe single path component."""


@dataclass(frozen=True)
class UploadLimits:
    max_file_bytes: int
    max_zip_total_bytes: int
    max_zip_members: int = MAX_ZIP_MEMBERS
    max_zip_ratio: float = MAX_ZIP_RATIO


def max_upload_bytes() -> int:
    raw = os.environ.get("UAV_MAX_UPLOAD_MB")
    try:
        mb = float(raw) if raw else float(DEFAULT_MAX_UPLOAD_MB)
    except ValueError:
        mb = float(DEFAULT_MAX_UPLOAD_MB)
    if mb <= 0:
        mb = float(DEFAULT_MAX_UPLOAD_MB)
    return int(mb * 1024 * 1024)


def get_limits() -> UploadLimits:
    per_file = max_upload_bytes()
    return UploadLimits(max_file_bytes=per_file, max_zip_total_bytes=per_file * 3)


def validate_flight_id(flight_id: str) -> str:
    """Return ``flight_id`` if it is a safe single path component, else raise."""
    if not isinstance(flight_id, str) or not FLIGHT_ID_RE.match(flight_id) or ".." in flight_id:
        raise InvalidFlightId(
            "Flight id must be 1-64 characters: letters, digits, '.', '_' or '-', "
            "starting with a letter or digit."
        )
    return flight_id
