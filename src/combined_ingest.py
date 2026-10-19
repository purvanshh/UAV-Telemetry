"""Combined-telemetry CSV: schema inspection, mapping, validation and normalization.

A *combined* file carries several signal groups (IMU / ATT / BARO) as columns of one
CSV on a shared time column. This module never scores anything. It

1. inspects the file (headers, samples, duplicate headers),
2. builds a **versioned mapping** from source columns to the *existing* canonical schema
   (``src/schema.py`` is the source of truth for names, groups, units and which fields
   are required),
3. validates it, and
4. splits the file into the same ``{"IMU","ATT","BARO"}`` frames the three-file upload
   produces, so ``src/validation.py``, ``src/timebase.py`` and the whole preprocessing
   and detector pipeline are reused unchanged. There is no second preprocessing path.

Mapping priority (see :func:`propose_mapping`):

1. **Deterministic** – a header equal (ignoring case, spaces, ``_``) to a documented
   dataset column (``GyrX``, ``Roll`` ...) or canonical feature name (``gyro_x`` ...).
   Their units are fixed by the dataset specification.
2. **LLM proposal** (``src/schema_llm.py``) – for unfamiliar headers only, with explicit
   consent. A proposal is untrusted input: it is re-validated here and is always left
   as ``needs_confirmation``.
3. **User confirmation** – the only way a non-deterministic mapping becomes ``valid``.
   The user states the canonical signal *and* the unit of the column.

Things this module deliberately does *not* do: guess units from a header suffix
(``phi_deg`` only produces a hint), treat radio/pilot inputs (``rc_*``) as desired
attitude, invent yaw, or convert units except through the small exact table below, and
only for a unit the user stated.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.dataset_profiles import Profile, load_profiles, match_profile
from src.limits import MAX_CSV_COLUMNS, MAX_CSV_ROWS, get_limits
from src.schema import STREAM_SCHEMA, STREAMS
from src.timebase import PLAUSIBLE_HZ, normalize_timestamps
from src.upload import UploadResult
from src.validation import MAX_NON_NUMERIC_FRAC, validate_flight

MAPPING_VERSION = "1.0"
TIME_TARGET = "__time__"

# validation_status values
VALID = "valid"
NEEDS_CONFIRMATION = "needs_confirmation"
REJECTED = "rejected"
AMBIGUOUS = "ambiguous"
UNMAPPED = "unmapped"

# unit_status values: only the first two are *verified*.
UNIT_DATASET = "dataset_spec"
UNIT_USER = "user_confirmed"
UNIT_PROFILE = "profile"  # stated by a dataset profile (explicit dataset configuration)
UNIT_HINT = "header_hint"  # inferred from the header text only -> never trusted
UNIT_LLM = "llm_stated"  # stated by a model -> never trusted
UNIT_UNKNOWN = "unknown"
VERIFIED_UNIT = (UNIT_DATASET, UNIT_PROFILE, UNIT_USER)


# ---- canonical targets, derived from src/schema.py ---------------------------------------

@dataclass(frozen=True)
class Target:
    canonical: str
    group: str  # IMU | ATT | BARO
    source_name: str  # dataset column name used downstream (e.g. "GyrX")
    unit: str  # unit the detector pipeline expects
    required: bool
    label: str
    range_check: Optional[Tuple[float, float]] = None


LABELS = {
    "gyro_x": "angular rate about X (gyro X)", "gyro_y": "angular rate about Y (gyro Y)",
    "gyro_z": "angular rate about Z (gyro Z)", "acc_x": "acceleration along X",
    "acc_y": "acceleration along Y", "acc_z": "acceleration along Z",
    "roll": "measured roll angle", "pitch": "measured pitch angle", "yaw": "measured heading (yaw)",
    "des_roll": "desired (setpoint) roll angle", "des_pitch": "desired (setpoint) pitch angle",
    "des_yaw": "desired (setpoint) yaw", "alt": "barometric altitude", "alt_amsl": "altitude above sea level",
    "press": "air pressure", "temp": "air temperature",
}


def _build_targets() -> Dict[str, Target]:
    out: Dict[str, Target] = {}
    for group in STREAMS:
        spec = STREAM_SCHEMA[group]
        for required, key in ((True, "required_features"), (False, "optional_features")):
            for canon, src in spec.get(key, {}).items():
                out[canon] = Target(
                    canonical=canon, group=group, source_name=src, unit=spec["units"][canon],
                    required=required, label=LABELS.get(canon, canon),
                    range_check=spec.get("range_checks", {}).get(canon),
                )
    return out


TARGETS: Dict[str, Target] = _build_targets()


def _norm(s: str) -> str:
    return re.sub(r"[\s_]+", "", str(s)).lower()


# Documented deterministic aliases: dataset column names and canonical feature names.
ALIASES: Dict[str, str] = {}
for _t in TARGETS.values():
    ALIASES[_norm(_t.source_name)] = _t.canonical
    ALIASES[_norm(_t.canonical)] = _t.canonical

# Time columns the existing timebase already supports (unit known from the dataset spec).
DETERMINISTIC_TIME = {"timeus": ("us", UNIT_DATASET), "timestamp": ("us", UNIT_DATASET)}
# Header suffix -> unit *hint*. Never sufficient on its own.
TIME_HINTS = {
    "tms": "ms", "timems": "ms", "ts": "s", "tsec": "s", "times": "s", "timesec": "s",
    "tus": "us", "timeusec": "us", "tns": "ns", "timens": "ns",
}
TIME_SCALE_TO_US = {"us": 1.0, "ms": 1e3, "s": 1e6, "ns": 1e-3}

# ---- units: exact conversions only ------------------------------------------------------

_UNIT_SYNONYMS = {
    "rad/s": "rad/s", "rads": "rad/s", "deg/s": "deg/s", "degs": "deg/s", "dps": "deg/s", "°/s": "deg/s",
    "m/s^2": "m/s^2", "m/s2": "m/s^2", "m/s²": "m/s^2", "ms2": "m/s^2", "g": "g",
    "deg": "deg", "degree": "deg", "degrees": "deg", "°": "deg", "rad": "rad", "radian": "rad", "radians": "rad",
    "mg": "mg", "m": "m", "meter": "m", "metre": "m", "meters": "m", "ft": "ft", "feet": "ft", "cm": "cm", "mm": "mm",
    "pa": "Pa", "hpa": "hPa", "mbar": "hPa", "kpa": "kPa", "bar": "bar",
    "degc": "degC", "°c": "degC", "c": "degC", "k": "K", "degf": "degF", "°f": "degF",
    "s": "s", "ms": "ms", "us": "us", "ns": "ns",
}
# (from, to) -> (scale, offset): to = from*scale + offset
_CONVERSIONS: Dict[Tuple[str, str], Tuple[float, float]] = {
    ("deg/s", "rad/s"): (math.pi / 180.0, 0.0),
    ("mg", "m/s^2"): (9.80665e-3, 0.0),
    ("g", "m/s^2"): (9.80665, 0.0),
    ("rad", "deg"): (180.0 / math.pi, 0.0),
    ("cm", "m"): (0.01, 0.0), ("mm", "m"): (0.001, 0.0), ("ft", "m"): (0.3048, 0.0),
    ("hPa", "Pa"): (100.0, 0.0), ("kPa", "Pa"): (1000.0, 0.0), ("bar", "Pa"): (1e5, 0.0),
    ("K", "degC"): (1.0, -273.15), ("degF", "degC"): (5.0 / 9.0, -160.0 / 9.0),
}


def normalize_unit(u: Optional[str]) -> Optional[str]:
    if u is None:
        return None
    key = str(u).strip().lower().replace(" ", "")
    return _UNIT_SYNONYMS.get(key) or _UNIT_SYNONYMS.get(key.replace("°", "deg"))


def conversion(from_unit: str, to_unit: str) -> Optional[Tuple[float, float]]:
    if from_unit == to_unit:
        return (1.0, 0.0)
    return _CONVERSIONS.get((from_unit, to_unit))


def unit_hint_from_header(header: str) -> Optional[str]:
    """Unit suggested by a header suffix such as ``phi_deg``. A hint, never evidence."""
    m = re.search(r"[_\s\[\(]([a-z/°^0-9]+)[\]\)]?$", str(header).strip().lower())
    if not m:
        return None
    return normalize_unit(m.group(1))


# ---- semantic notes: why some obvious-looking columns must not be mapped -----------------

_RC_RE = re.compile(r"^rc[_\s-]?", re.I)
_FLAG_RE = re.compile(r"(^armed$|^arm$|failsafe|(^|_)fs$|(^|_)ok$|^mode$)", re.I)


def semantic_note(header: str) -> Optional[str]:
    if _RC_RE.match(header):
        return (
            "Radio/pilot input channel. Stick commands are not the aircraft's desired attitude "
            "and are never used as desired roll or pitch."
        )
    if _FLAG_RE.search(header):
        return "Looks like a status flag rather than a measured signal; the detector does not use it."
    return None


def rc_blocked(source: str, canonical: str) -> bool:
    return bool(_RC_RE.match(source)) and canonical.startswith("des_")


# ---- data classes ------------------------------------------------------------------------


@dataclass
class ColumnMapping:
    source_column: str
    canonical: Optional[str] = None  # canonical signal name, TIME_TARGET, or None
    group: Optional[str] = None
    unit: Optional[str] = None  # unit of the SOURCE column, if known
    expected_unit: Optional[str] = None
    unit_status: str = UNIT_UNKNOWN
    required: bool = False
    method: str = "none"  # deterministic_alias | llm_proposed | user_confirmed | none
    validation_status: str = UNMAPPED
    ambiguity: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    scale: float = 1.0
    offset: float = 0.0
    llm_confidence: Optional[str] = None
    unresolved: Optional[str] = None  # "unit" | "meaning": the one fact still missing
    semantics: Optional[str] = None  # where the signal meaning came from
    profile_id: Optional[str] = None
    ignored: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_column": self.source_column, "canonical": self.canonical, "group": self.group,
            "unit": self.unit, "expected_unit": self.expected_unit, "unit_status": self.unit_status,
            "required": self.required, "method": self.method, "validation_status": self.validation_status,
            "ambiguity": self.ambiguity, "notes": list(self.notes), "llm_confidence": self.llm_confidence,
            "unresolved": self.unresolved, "semantics": self.semantics, "profile_id": self.profile_id,
            "ignored": self.ignored,
            "conversion": None if (self.scale, self.offset) == (1.0, 0.0)
            else {"scale": self.scale, "offset": self.offset},
        }


@dataclass
class SchemaMapping:
    version: str = MAPPING_VERSION
    schema_fingerprint: str = ""
    columns: List[ColumnMapping] = field(default_factory=list)
    time: Optional[ColumnMapping] = None
    missing_required: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    profile_id: Optional[str] = None

    @property
    def scoreable(self) -> bool:
        return not self.blockers

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version, "schema_fingerprint": self.schema_fingerprint, "profile_id": self.profile_id,
            "time": self.time.to_dict() if self.time else None,
            "columns": [c.to_dict() for c in self.columns],
            "missing_required": list(self.missing_required),
            "missing_required_labels": [TARGETS[m].label for m in self.missing_required],
            "blockers": list(self.blockers), "warnings": list(self.warnings),
            "scoreable": self.scoreable,
        }


# ---- inspection ---------------------------------------------------------------------------


class CombinedFileError(ValueError):
    """The file cannot be inspected at all (unreadable, empty, duplicate headers ...)."""


@dataclass
class Inspection:
    filename: str
    size_bytes: int
    sha256: str
    df: pd.DataFrame
    headers: List[str]
    stats: Dict[str, Dict[str, Any]]  # per column: numeric/nan/text fractions, samples

    def preview_rows(self, n: int = 5) -> List[Dict[str, Any]]:
        head = self.df.head(n).astype(object)
        return head.where(pd.notnull(head), None).to_dict(orient="records")


def _read_header(raw: bytes) -> List[str]:
    text = raw[: 1 << 20].decode("utf-8-sig", errors="replace")
    try:
        row = next(csv.reader(io.StringIO(text)))
    except StopIteration:
        raise CombinedFileError("The file is empty.")
    return [h.strip() for h in row]


def column_stats(s: pd.Series) -> Dict[str, Any]:
    n = len(s)
    nonnull = s.notna()
    num = pd.to_numeric(s, errors="coerce")
    arr = num.to_numpy(dtype="float64", na_value=np.nan)
    finite = np.isfinite(arr)
    n_text = int((nonnull & num.isna()).sum())
    n_inf = int(np.isinf(arr).sum())
    samples: List[Any] = []
    for v in s.dropna().head(5).tolist():
        if isinstance(v, str):
            samples.append("".join(ch for ch in v[:24] if ch.isprintable()))
        else:
            samples.append(float(f"{float(v):.6g}"))
    return {
        "n": n, "n_missing": int((~nonnull).sum()), "n_text": n_text, "n_inf": n_inf,
        "n_finite": int(finite.sum()),
        "numeric_fraction": float(finite.sum() / max(int(nonnull.sum()), 1)),
        "min": float(arr[finite].min()) if finite.any() else None,
        "max": float(arr[finite].max()) if finite.any() else None,
        "samples": samples,
    }


def inspect_combined(raw: bytes, filename: str = "combined.csv") -> Inspection:
    """Parse headers and data with all resource limits applied. Raises CombinedFileError."""
    lim = get_limits()
    if len(raw) == 0:
        raise CombinedFileError("The file is empty.")
    if len(raw) > lim.max_file_bytes:
        raise CombinedFileError(
            f"'{filename}' is {len(raw) / 1e6:.1f} MB, over the {lim.max_file_bytes / 1e6:.0f} MB upload limit."
        )
    headers = _read_header(raw)
    if len(headers) > MAX_CSV_COLUMNS:
        raise CombinedFileError(f"The file has {len(headers)} columns; the limit is {MAX_CSV_COLUMNS}.")
    if any(h == "" for h in headers):
        raise CombinedFileError("The header row contains an empty column name.")
    seen: Dict[str, int] = {}
    for h in headers:
        seen[_norm(h)] = seen.get(_norm(h), 0) + 1
    dups = sorted({h for h in headers if seen[_norm(h)] > 1})
    if dups:
        raise CombinedFileError(
            "Duplicate column names (case, spaces and underscores ignored): "
            + ", ".join(dups) + ". Rename them so every column is unique."
        )
    try:
        df = pd.read_csv(io.BytesIO(raw), nrows=MAX_CSV_ROWS + 1, encoding="utf-8-sig")
    except Exception as exc:  # parser errors are user-facing
        raise CombinedFileError(f"Could not read '{filename}' as CSV: {exc}") from exc
    if df.empty:
        raise CombinedFileError("The file has a header but no data rows.")
    if len(df) > MAX_CSV_ROWS:
        raise CombinedFileError(f"The file has more than {MAX_CSV_ROWS:,} rows.")
    df.columns = [str(c).strip() for c in df.columns]
    return Inspection(
        filename=filename, size_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), df=df,
        headers=list(df.columns), stats={c: column_stats(df[c]) for c in df.columns},
    )


def schema_fingerprint(headers: List[str]) -> str:
    return hashlib.sha256("|".join(sorted(_norm(h) for h in headers)).encode()).hexdigest()[:16]


# ---- mapping ------------------------------------------------------------------------------


def _finalize_numeric_check(cm: ColumnMapping, st: Dict[str, Any]) -> None:
    bad = st["n_text"] + st["n_inf"]
    nonnull = st["n"] - st["n_missing"]
    if st["n_finite"] == 0:
        cm.validation_status, cm.ambiguity = REJECTED, "Column has no numeric values."
    elif nonnull and bad / nonnull > MAX_NON_NUMERIC_FRAC:
        cm.validation_status = REJECTED
        cm.ambiguity = f"{bad} of {nonnull} values are text or infinite."


def _range_note(cm: ColumnMapping, st: Dict[str, Any], tgt: Target) -> None:
    if tgt.range_check is None or st["min"] is None:
        return
    lo, hi = tgt.range_check
    a, b = st["min"] * cm.scale + cm.offset, st["max"] * cm.scale + cm.offset
    if a < lo or b > hi:
        cm.notes.append(
            f"After conversion the values span {a:.3g}..{b:.3g} {tgt.unit}, outside the plausible "
            f"{lo:g}..{hi:g}. Check that the unit is right."
        )


def _time_mapping(
    insp: Inspection, assignments: Dict[str, Dict[str, Any]], profile: Optional[Profile]
) -> Optional[ColumnMapping]:
    """Pick the time column: dataset names, then a user choice, then a profile, then a hint."""
    for h in insp.headers:
        key = _norm(h)
        if key in DETERMINISTIC_TIME and h not in assignments:
            unit, ustat = DETERMINISTIC_TIME[key]
            return ColumnMapping(
                source_column=h, canonical=TIME_TARGET, unit=unit, expected_unit="us", unit_status=ustat,
                required=True, method="deterministic_alias", validation_status=VALID, semantics="dataset_name",
            )
    for h, a in assignments.items():
        if a.get("canonical") == TIME_TARGET and h in insp.headers:
            u = normalize_unit(a.get("unit"))
            cm = ColumnMapping(source_column=h, canonical=TIME_TARGET, expected_unit="us", required=True,
                               method="user_confirmed", semantics="user")
            if u not in TIME_SCALE_TO_US:
                cm.validation_status, cm.ambiguity, cm.unresolved = REJECTED, "Choose the time unit: s, ms, us or ns.", "unit"
                return cm
            cm.unit, cm.unit_status, cm.validation_status = u, UNIT_USER, VALID
            return cm
    if profile and profile.time_column and profile.time_unit:
        col = next((h for h in insp.headers if _norm(h) == _norm(profile.time_column)), None)
        if col:
            return ColumnMapping(
                source_column=col, canonical=TIME_TARGET, unit=profile.time_unit, expected_unit="us",
                unit_status=UNIT_PROFILE, required=True, method="dataset_profile", validation_status=VALID,
                semantics="profile", profile_id=profile.id,
            )
    for h in insp.headers:
        hint = TIME_HINTS.get(_norm(h))
        if hint:
            return ColumnMapping(
                source_column=h, canonical=TIME_TARGET, unit=hint, expected_unit="us", unit_status=UNIT_HINT,
                required=True, method="none", validation_status=NEEDS_CONFIRMATION, unresolved="unit",
                ambiguity=f"'{h}' looks like a time column, but its unit ({hint}) is only suggested by the name "
                "and no dataset definition states it. Confirm the time unit.",
            )
    return None


def _apply_unit(cm: ColumnMapping, unit: Optional[str], status: str) -> None:
    """Attach a *verified* unit and its exact conversion, or reject if no exact conversion exists."""
    tgt = TARGETS[cm.canonical]
    conv = conversion(unit, tgt.unit) if unit else None
    if conv is None:
        cm.unit, cm.validation_status = unit, REJECTED
        cm.ambiguity = f"Unit '{unit}' cannot be converted exactly to {tgt.unit}."
        return
    cm.unit, cm.unit_status, cm.scale, cm.offset = unit, status, conv[0], conv[1]
    cm.validation_status, cm.unresolved, cm.ambiguity = VALID, None, None


def _plausible_in_some_unit(canon: str, st: Dict[str, Any]) -> bool:
    """True if the observed range fits the signal's plausible range under at least one exactly
    convertible unit. A range test can *reject* a proposal; it never proves a unit."""
    t = TARGETS[canon]
    if t.range_check is None or st["min"] is None:
        return st["n_finite"] > 0
    lo, hi = t.range_check
    units = {t.unit} | {u for (u, v) in _CONVERSIONS if v == t.unit}
    for u in units:
        sc, off = conversion(u, t.unit) or (1.0, 0.0)
        a, b = st["min"] * sc + off, st["max"] * sc + off
        if lo <= min(a, b) and max(a, b) <= hi:
            return True
    return False


def propose_mapping(
    insp: Inspection,
    *,
    assignments: Optional[Dict[str, Dict[str, Any]]] = None,
    llm_proposals: Optional[List[Dict[str, Any]]] = None,
    profile: Optional[Profile] = None,
) -> SchemaMapping:
    """Build the mapping. Precedence per column: user assignment, dataset name/alias, dataset
    profile, validated AI proposal, otherwise unused.

    * ``valid`` needs the signal meaning AND a verified unit (dataset definition, profile or user).
    * AI proposals can establish *meaning* only when they pass local checks; their unit is never
      trusted, so they end as ``needs_confirmation`` with ``unresolved="unit"``.
    * ``assignments`` is explicit user input: ``{"col": {"canonical"?: ..., "unit": ...}}``; the
      canonical may be omitted to just supply the unit for an already-proposed meaning.
    """
    assignments = dict(assignments or {})
    m = SchemaMapping(schema_fingerprint=schema_fingerprint(insp.headers), profile_id=profile.id if profile else None)
    m.time = _time_mapping(insp, assignments, profile)
    time_col = m.time.source_column if m.time else None
    llm_by_col = {p["source_column"]: p for p in (llm_proposals or []) if p.get("canonical")}
    claimed: Dict[str, List[ColumnMapping]] = {}
    # Signals already established without AI. A model suggestion may never compete with them.
    taken = {ALIASES[_norm(h)] for h in insp.headers if _norm(h) in ALIASES}
    if profile:
        taken |= {pc.canonical for h in insp.headers if (pc := profile.column_for(h)) is not None}
    taken |= {a["canonical"] for a in assignments.values() if a.get("canonical") in TARGETS}

    for h in insp.headers:
        if h == time_col:
            continue
        st = insp.stats[h]
        a = assignments.get(h)
        alias = ALIASES.get(_norm(h))
        pcol = profile.column_for(h) if profile else None

        # 1. base proposal (without user input)
        if alias:
            t = TARGETS[alias]
            base = ColumnMapping(
                source_column=h, canonical=alias, group=t.group, expected_unit=t.unit, required=t.required,
                method="deterministic_alias", semantics="dataset_name",
            )
            _apply_unit(base, t.unit, UNIT_DATASET)
        elif pcol:
            t = TARGETS[pcol.canonical]
            base = ColumnMapping(
                source_column=h, canonical=pcol.canonical, group=t.group, expected_unit=t.unit, required=t.required,
                method="dataset_profile", semantics="profile", profile_id=profile.id,
            )
            if pcol.unit:
                _apply_unit(base, pcol.unit, UNIT_PROFILE)
            else:
                base.validation_status, base.unresolved = NEEDS_CONFIRMATION, "unit"
                base.ambiguity = pcol.unresolved or "The unit of this column is not documented."
        elif profile and profile.is_ignored(h):
            base = ColumnMapping(source_column=h, ignored=True, method="dataset_profile", profile_id=profile.id)
            base.notes = ["Not used by the deployed detector."]
        elif h in llm_by_col and llm_by_col[h].get("canonical") not in taken:
            base = _llm_mapping(h, llm_by_col[h], st)
        else:
            base = ColumnMapping(source_column=h)
            base.notes = [n for n in (semantic_note(h),) if n]
            hint = unit_hint_from_header(h)
            if hint:
                base.unit, base.unit_status = hint, UNIT_HINT
                base.notes.append(f"Header suggests unit '{hint}' (not verified, not used).")

        # 2. explicit user input overrides / completes the proposal
        cm = _user_mapping(h, a, st, base) if a else base
        if cm.canonical and cm.validation_status == VALID:
            _finalize_numeric_check(cm, st)
        if cm.canonical and cm.validation_status in (VALID, NEEDS_CONFIRMATION):
            claimed.setdefault(cm.canonical, []).append(cm)
        m.columns.append(cm)

    # two columns claiming one signal -> ambiguity, neither is used
    for canon, cms in claimed.items():
        if len(cms) > 1:
            names = ", ".join(c.source_column for c in cms)
            for c in cms:
                c.validation_status = AMBIGUOUS
                c.unresolved = "meaning"
                c.ambiguity = f"{names} all map to '{canon}'. Keep exactly one."

    by_canon = {c.canonical: c for c in m.columns if c.canonical and c.validation_status == VALID}
    for t in TARGETS.values():
        if t.required and t.canonical not in by_canon:
            m.missing_required.append(t.canonical)
    for c in m.columns:
        if c.canonical:
            _range_note(c, insp.stats[c.source_column], TARGETS[c.canonical])
            st = insp.stats[c.source_column]
            bad = st["n_text"] + st["n_inf"]
            if bad and c.validation_status == VALID:
                m.warnings.append(
                    f"{c.source_column}: {bad} non-numeric or infinite value(s) are treated as missing."
                )
    _compute_blockers(m, insp)
    return m


def _user_mapping(h: str, a: Dict[str, Any], st: Dict[str, Any], base: ColumnMapping) -> ColumnMapping:
    canon = a.get("canonical") or base.canonical
    cm = ColumnMapping(source_column=h, method="user_confirmed", semantics="user")
    if canon is None:
        # A unit with no known meaning yet: keep the column as it was (the AI step may still name it).
        base.notes = base.notes + ["A unit was given but the column's meaning is not known yet."]
        return base
    if canon not in TARGETS:
        cm.validation_status = REJECTED
        cm.ambiguity = f"'{canon}' is not a signal the detector pipeline knows." if canon else "Choose what this column is."
        return cm
    t = TARGETS[canon]
    cm.canonical, cm.group, cm.expected_unit, cm.required = canon, t.group, t.unit, t.required
    if rc_blocked(h, canon):
        cm.canonical = None
        cm.validation_status = REJECTED
        cm.ambiguity = (
            f"'{h}' is a radio/pilot input and cannot be used as '{t.label}'. Provide a column "
            "holding the aircraft's desired attitude in degrees."
        )
        return cm
    # A column whose unit a dataset definition/profile already established keeps it; the user
    # cannot silently override it by omission, but an explicit different unit is taken as theirs.
    unit = normalize_unit(a.get("unit"))
    if unit is None:
        if base.canonical == canon and base.validation_status == VALID:
            return base
        cm.validation_status, cm.unresolved = NEEDS_CONFIRMATION, "unit"
        cm.ambiguity = f"State the unit of '{h}' (expected {t.unit} or an exactly convertible unit)."
        return cm
    conv = conversion(unit, t.unit)
    if conv is None:
        cm.unit, cm.validation_status, cm.unresolved = unit, REJECTED, "unit"
        cm.ambiguity = f"Unit '{unit}' cannot be converted exactly to {t.unit}."
        return cm
    _apply_unit(cm, unit, UNIT_USER)
    return cm


def _llm_mapping(h: str, p: Dict[str, Any], st: Dict[str, Any]) -> ColumnMapping:
    """Turn a model suggestion into a mapping, applying the local acceptance criteria.

    Meaning is accepted (``semantics="ai_validated"``) only if: confidence is medium/high, the
    target is a real signal and not a forbidden pairing, the column is numeric, and its values are
    plausible for the signal under some exactly-convertible unit. The unit is *never* taken from
    the model: the column stays ``needs_confirmation`` until a dataset definition, a profile or the
    user supplies it.
    """
    canon = p.get("canonical")
    if canon not in TARGETS or rc_blocked(h, canon):
        return ColumnMapping(source_column=h)
    t = TARGETS[canon]
    cm = ColumnMapping(source_column=h, method="llm_proposed", llm_confidence=p.get("confidence"),
                       canonical=canon, group=t.group, expected_unit=t.unit, required=t.required)
    cm.unit, cm.unit_status = normalize_unit(p.get("unit")), UNIT_LLM
    if st["numeric_fraction"] < 0.99 or st["n_finite"] == 0:
        cm.validation_status, cm.ambiguity, cm.canonical = REJECTED, "AI suggestion rejected: column is not numeric.", None
        return cm
    if p.get("confidence") not in ("medium", "high"):
        cm.validation_status, cm.unresolved = NEEDS_CONFIRMATION, "meaning"
        cm.ambiguity = "The AI suggestion for this column was low-confidence. Confirm what it is and its unit."
        return cm
    if not _plausible_in_some_unit(canon, st):
        cm.validation_status, cm.canonical, cm.unresolved = REJECTED, None, "meaning"
        cm.ambiguity = f"AI suggestion rejected: the values are implausible for {t.label} in any supported unit."
        return cm
    cm.semantics = "ai_validated"
    cm.validation_status, cm.unresolved = NEEDS_CONFIRMATION, "unit"
    cm.ambiguity = (
        f"Meaning ({t.label}) suggested by AI and checked against the data, but the unit is not established. "
        "State the unit."
    )
    return cm


def _compute_blockers(m: SchemaMapping, insp: Inspection) -> None:
    b = m.blockers
    if m.time is None:
        b.append("No time column was found. Expected TimeUS or Timestamp, or a time column whose unit is documented.")
    elif m.time.validation_status != VALID:
        b.append(m.time.ambiguity or "The time column's unit is not established.")
    for c in m.columns:
        if c.validation_status == AMBIGUOUS:
            b.append(c.ambiguity or f"{c.source_column} is ambiguous.")
        elif c.required and c.validation_status in (NEEDS_CONFIRMATION, REJECTED) and c.canonical:
            b.append(f"{c.source_column}: {c.ambiguity}")
        elif c.method == "llm_proposed" and c.validation_status == NEEDS_CONFIRMATION:
            m.warnings.append(f"{c.source_column}: AI suggestion for '{c.canonical}' is not applied until its unit is stated.")
    if m.missing_required:
        b.append(
            "The deployed detector needs signals this file does not provide or has not resolved: "
            + ", ".join(f"{TARGETS[k].label} ({k})" for k in m.missing_required) + "."
        )


# ---- timestamp validation (reuses src/timebase.py) -------------------------------------------


def time_microseconds(insp: Inspection, tm: ColumnMapping) -> pd.Series:
    scale = TIME_SCALE_TO_US[tm.unit or "us"]
    return pd.to_numeric(insp.df[tm.source_column], errors="coerce") * scale


def check_timestamps(insp: Inspection, tm: Optional[ColumnMapping]) -> Dict[str, Any]:
    """Run the shared timebase checks on the candidate time column + unit."""
    if tm is None or tm.unit not in TIME_SCALE_TO_US:
        return {"checked": False}
    trial = pd.DataFrame({"TimeUS": time_microseconds(insp, tm)})
    _clean, rep = normalize_timestamps(trial, stream="IMU")
    return {
        "checked": True, "unit": tm.unit, "unit_status": tm.unit_status, "n_rows": rep.n_rows,
        "n_missing": rep.n_missing, "n_duplicates": rep.n_duplicates,
        "n_nonmonotonic_steps": rep.n_nonmonotonic_steps, "median_dt_s": rep.median_dt_s,
        "est_hz": rep.est_hz, "duration_s": rep.duration_s, "errors": list(rep.errors),
        "warnings": list(rep.warnings), "plausible_hz": list(PLAUSIBLE_HZ),
    }


# ---- normalization into the existing three-stream representation -----------------------------


def to_stream_frames(insp: Inspection, mapping: SchemaMapping) -> Dict[str, pd.DataFrame]:
    """Split the combined table into IMU / ATT / BARO frames using dataset column names.

    Only ``valid`` mappings are used; values are converted with the mapping's exact
    scale/offset (identity for dataset-spec columns). A stream keeps a row only if at least
    one of its columns has a value, so a combined file whose groups sample at different
    rates splits back into the original per-stream rows.
    """
    if mapping.time is None or mapping.time.validation_status != VALID:
        raise CombinedFileError("The time column is not confirmed.")
    t_us = time_microseconds(insp, mapping.time)
    used = [c for c in mapping.columns if c.canonical and c.validation_status == VALID]
    frames: Dict[str, pd.DataFrame] = {}
    for group in STREAMS:
        cols = {"TimeUS": t_us}
        data_names = []
        for c in used:
            if c.group != group:
                continue
            v = pd.to_numeric(insp.df[c.source_column], errors="coerce").astype("float64")
            v = v.where(np.isfinite(v))
            if (c.scale, c.offset) != (1.0, 0.0):
                v = v * c.scale + c.offset
            name = TARGETS[c.canonical].source_name
            cols[name] = v
            data_names.append(name)
        f = pd.DataFrame(cols)
        if data_names:
            f = f[f[data_names].notna().any(axis=1)]
        frames[group] = f.reset_index(drop=True)
    return frames


def mapping_for_storage(
    mapping: SchemaMapping, insp: Inspection, llm: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Provenance record kept in meta.json: mapping, file identity, no secrets, no data."""
    return {
        "kind": "combined_csv", "mapping_version": mapping.version,
        "schema_fingerprint": mapping.schema_fingerprint, "filename": insp.filename,
        "sha256": insp.sha256, "size_bytes": insp.size_bytes, "rows": int(len(insp.df)),
        "mapping": mapping.to_dict(),
        "llm": {
            "used_for_columns": [c.source_column for c in mapping.columns if c.method == "llm_proposed"],
            **({k: llm[k] for k in ("status", "model", "prompt_version", "columns_sent") if k in llm} if llm else {}),
        },
        "profile_id": mapping.profile_id,
    }


def load_combined(
    raw: bytes,
    filename: str,
    *,
    assignments: Optional[Dict[str, Dict[str, Any]]] = None,
    llm_proposals: Optional[List[Dict[str, Any]]] = None,
    profile: Optional[Profile] = None,
    use_profiles: bool = True,
) -> Tuple[UploadResult, Optional[Inspection], Optional[SchemaMapping]]:
    """Full path from bytes to an :class:`UploadResult` in the shape the 3-file loader returns."""
    res = UploadResult(ok=True, source="combined")
    try:
        insp = inspect_combined(raw, filename)
    except CombinedFileError as exc:
        res.ok = False
        res.errors.append(str(exc))
        return res, None, None
    res.filenames = {s: filename for s in STREAMS}
    res.file_sizes = {s: len(raw) for s in STREAMS}
    res.file_sha256 = {s: insp.sha256 for s in STREAMS}

    if profile is None and use_profiles:
        profile = match_profile(insp.headers)
    mapping = propose_mapping(insp, assignments=assignments, llm_proposals=llm_proposals, profile=profile)
    res.warnings.extend(mapping.warnings)
    if not mapping.scoreable:
        res.ok = False
        res.errors.extend(mapping.blockers)
        return res, insp, mapping

    ts = check_timestamps(insp, mapping.time)
    if ts.get("errors"):
        res.ok = False
        res.errors.extend(ts["errors"])
        return res, insp, mapping

    res.frames = to_stream_frames(insp, mapping)
    # Everything below is the existing validator: required columns, timestamps, ranges.
    res.validation = validate_flight(res.frames, filenames=res.filenames, file_sizes=res.file_sizes)
    if not res.validation.ok:
        res.ok = False
    return res, insp, mapping
