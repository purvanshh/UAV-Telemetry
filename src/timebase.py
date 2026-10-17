"""Single source of truth for timestamp interpretation.

Both upload validation (``src/validation.py``) and preprocessing
(``src/preprocessing_adapter.py``) call :func:`normalize_timestamps`, so the two
stages can no longer disagree about units.

Supported timestamp fields (priority order, matches ``uav_model/preprocessing.py``):

======================  =====  ============================================
Column                  Unit   Basis
======================  =====  ============================================
``TimeUS``              µs     explicit — the unit is part of the field name
                               (ArduPilot DataFlash convention)
``Timestamp`` (numeric) µs     ``schema_default`` — ``DEFAULT_SCHEMA`` in
                               ``uav_model/preprocessing.py`` declares
                               ``ts_unit="us"``; the file itself carries no
                               unit, so a plausibility check guards it
======================  =====  ============================================

The unit is **never** inferred from value magnitude. A ``Timestamp`` column whose
implied sampling rate is implausible under the declared unit (e.g. seconds-valued
data read as µs) is rejected as ``unit_ambiguous`` instead of being reinterpreted.
Datetime strings are not supported (they were accepted by the old validator but
could not be aligned) and are rejected with a clear message.

Semantics preserved from the previous ``load_stream_df``: seconds = raw / 1e6,
rows with unparsable timestamps are dropped, rows are sorted by time and exact
duplicate timestamps keep the first row. The only difference is that the sort is
now stable so "first" means first in file order.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# (column name, unit, unit_source) in priority order.
SUPPORTED_TIMESTAMP_FIELDS = (
    ("TimeUS", "us", "explicit_field_name"),
    ("Timestamp", "us", "schema_default"),
)
UNIT_DIVISOR = {"us": 1e6, "ms": 1e3, "s": 1.0, "ns": 1e9}

# Plausible per-stream sampling rates under the declared unit (Hz).
PLAUSIBLE_HZ = (0.05, 5000.0)
# Rejection thresholds for time-order problems.
MAX_MISSING_TIMESTAMP_FRAC = 0.5  # unchanged from the previous validator
MAX_NONMONOTONIC_FRAC = 0.05
MAX_BACKWARD_JUMP_S = 1.0  # a larger jump looks like a clock reset, not jitter


def _norm(s: str) -> str:
    return re.sub(r"[\s_]+", "", str(s)).lower()


def find_timestamp_column(columns: Sequence[str]) -> Optional[str]:
    lut = {_norm(c): c for c in columns}
    for name, _unit, _src in SUPPORTED_TIMESTAMP_FIELDS:
        if _norm(name) in lut:
            return lut[_norm(name)]
    return None


@dataclass
class TimebaseReport:
    """Everything known about one stream's time axis (counts are over raw rows)."""

    column: Optional[str] = None
    unit: Optional[str] = None
    unit_source: Optional[str] = None  # explicit_field_name | schema_default
    n_rows: int = 0
    n_missing: int = 0  # NaN / unparsable timestamp (row dropped)
    n_valid: int = 0  # rows with a finite timestamp
    n_duplicates: int = 0  # valid rows whose timestamp repeats an earlier one (dropped)
    n_unique: int = 0  # rows kept after de-duplication == n_valid - n_duplicates
    n_nonmonotonic_steps: int = 0  # file-order steps with dt < 0 among valid rows
    max_backward_jump_s: float = 0.0
    raw_first: Optional[float] = None  # first valid raw value, original unit, file order
    raw_last: Optional[float] = None
    t_start_s: Optional[float] = None
    t_end_s: Optional[float] = None
    duration_s: Optional[float] = None
    median_dt_s: Optional[float] = None
    est_hz: Optional[float] = None
    gaps: List[Dict[str, float]] = field(default_factory=list)  # {start_s,end_s,duration_s}
    max_gap_s: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    status: str = "ok"  # ok | unit_ambiguous | unsupported | unusable | time_order_ambiguous

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_timestamps(
    df: pd.DataFrame,
    *,
    stream: str = "",
    gap_threshold_s: float = 0.5,
) -> "tuple[pd.DataFrame, TimebaseReport]":
    """Return ``(clean_df, report)``.

    ``clean_df`` has a ``__t_s`` column (seconds, strictly increasing), is sorted
    and de-duplicated, and is empty when the report carries errors.
    ``gap_threshold_s`` is the preprocessing ``max_gap_s``: source gaps longer than
    this are listed because interpolation will not bridge them.
    """
    tag = f"{stream}: " if stream else ""
    rep = TimebaseReport(n_rows=int(len(df)))
    empty = df.iloc[0:0].copy()
    empty["__t_s"] = np.array([], dtype="float64")

    col = find_timestamp_column(list(df.columns))
    if col is None:
        rep.status = "unsupported"
        rep.errors.append(
            f"{tag}no usable timestamp column. Supported: "
            + ", ".join(n for n, _u, _s in SUPPORTED_TIMESTAMP_FIELDS)
            + "."
        )
        return empty, rep
    rep.column = col
    for name, unit, src in SUPPORTED_TIMESTAMP_FIELDS:
        if _norm(name) == _norm(col):
            rep.unit, rep.unit_source = unit, src
            break

    raw = pd.to_numeric(df[col], errors="coerce").astype("float64")
    finite = np.isfinite(raw.to_numpy())
    rep.n_missing = int((~finite).sum())
    rep.n_valid = int(finite.sum())

    if rep.n_valid < max(2, int((1 - MAX_MISSING_TIMESTAMP_FRAC) * rep.n_rows)):
        # Text that is not numeric (e.g. ISO datetimes) lands here.
        looks_dt = False
        try:
            parsed = pd.to_datetime(df[col], errors="coerce", utc=True)
            looks_dt = bool(parsed.notna().sum() >= max(2, rep.n_rows // 2))
        except Exception:
            pass
        rep.status = "unsupported" if looks_dt else "unusable"
        rep.errors.append(
            f"{tag}timestamp column '{col}' "
            + (
                "contains datetime strings; only numeric "
                f"{'/'.join(n for n, _u, _s in SUPPORTED_TIMESTAMP_FIELDS)} "
                "values are supported."
                if looks_dt
                else f"has too few usable values ({rep.n_valid}/{rep.n_rows})."
            )
        )
        return empty, rep

    if rep.n_missing:
        rep.warnings.append(
            f"{tag}{rep.n_missing} row(s) with missing/unparsable timestamps were dropped."
        )

    divisor = UNIT_DIVISOR[rep.unit or "us"]
    t_file = raw.to_numpy()[finite] / divisor  # file order, valid rows only
    rep.raw_first = float(raw.to_numpy()[finite][0])
    rep.raw_last = float(raw.to_numpy()[finite][-1])

    steps = np.diff(t_file)
    rep.n_nonmonotonic_steps = int((steps < 0).sum())
    rep.max_backward_jump_s = float(-steps.min()) if rep.n_nonmonotonic_steps else 0.0

    order = np.argsort(t_file, kind="stable")
    t_sorted = t_file[order]
    keep = np.ones(len(t_sorted), dtype=bool)
    keep[1:] = np.diff(t_sorted) != 0
    rep.n_duplicates = int((~keep).sum())
    rep.n_unique = int(keep.sum())
    t_unique = t_sorted[keep]

    if rep.n_unique < 2:
        rep.status = "unusable"
        rep.errors.append(f"{tag}fewer than 2 distinct timestamps.")
        return empty, rep

    # --- time-order sanity -------------------------------------------------
    if rep.max_backward_jump_s > MAX_BACKWARD_JUMP_S or (
        rep.n_nonmonotonic_steps / max(1, len(steps)) > MAX_NONMONOTONIC_FRAC
    ):
        rep.status = "time_order_ambiguous"
        rep.errors.append(
            f"{tag}timestamps are not monotonic ({rep.n_nonmonotonic_steps} backward "
            f"step(s), largest {rep.max_backward_jump_s:.3g} s). This looks like a "
            "clock reset or interleaved logs; sorting would silently mix timelines."
        )
        return empty, rep

    # --- unit plausibility (declared unit, never re-inferred) --------------
    dts = np.diff(t_unique)
    rep.median_dt_s = float(np.median(dts))
    rep.est_hz = 1.0 / rep.median_dt_s
    rep.t_start_s, rep.t_end_s = float(t_unique[0]), float(t_unique[-1])
    rep.duration_s = rep.t_end_s - rep.t_start_s
    lo, hi = PLAUSIBLE_HZ
    if not (lo <= rep.est_hz <= hi):
        rep.status = "unit_ambiguous"
        rep.errors.append(
            f"{tag}timestamp unit is ambiguous: reading '{col}' as {rep.unit} gives "
            f"an implied sampling rate of {rep.est_hz:.4g} Hz (plausible range "
            f"{lo}–{hi} Hz). The unit is not guessed from magnitude — supply "
            "TimeUS (microseconds) or correct the column."
        )
        return empty, rep
    if rep.unit_source == "schema_default":
        rep.warnings.append(
            f"{tag}'{col}' carries no unit; interpreted as {rep.unit} per the "
            "deployed preprocessing schema (implied "
            f"{rep.est_hz:.3g} Hz is plausible). Prefer a TimeUS column."
        )

    # --- gaps (informational: interpolation will not bridge them) ----------
    big = np.where(dts > gap_threshold_s)[0]
    rep.gaps = [
        {
            "start_s": float(t_unique[i]),
            "end_s": float(t_unique[i + 1]),
            "duration_s": float(dts[i]),
        }
        for i in big
    ]
    rep.max_gap_s = float(dts.max())
    if rep.n_nonmonotonic_steps:
        rep.warnings.append(
            f"{tag}{rep.n_nonmonotonic_steps} non-monotonic timestamp step(s) in file "
            "order (rows are sorted by time before analysis)."
        )
    if rep.n_duplicates:
        rep.warnings.append(
            f"{tag}{rep.n_duplicates} duplicate timestamp(s); the first row for each is kept."
        )
    if rep.gaps:
        rep.warnings.append(
            f"{tag}{len(rep.gaps)} source gap(s) longer than {gap_threshold_s:g} s "
            f"(longest {rep.max_gap_s:.3g} s); values inside gaps are not interpolated."
        )

    # --- build the cleaned frame (same semantics as load_stream_df) --------
    valid_idx = np.flatnonzero(finite)
    rows = valid_idx[order][keep]
    out = df.iloc[rows].copy()
    out["__t_s"] = t_unique
    out = out.reset_index(drop=True)
    return out, rep
