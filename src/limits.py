"""Upload size limit. ``UAV_MAX_UPLOAD_MB`` overrides the per-file cap (default 100 MB)."""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_MAX_UPLOAD_MB = 100


@dataclass(frozen=True)
class UploadLimits:
    max_file_bytes: int


def get_limits() -> UploadLimits:
    raw = os.environ.get("UAV_MAX_UPLOAD_MB")
    try:
        mb = float(raw) if raw else float(DEFAULT_MAX_UPLOAD_MB)
    except ValueError:
        mb = float(DEFAULT_MAX_UPLOAD_MB)
    if mb <= 0:
        mb = float(DEFAULT_MAX_UPLOAD_MB)
    return UploadLimits(max_file_bytes=int(mb * 1024 * 1024))
