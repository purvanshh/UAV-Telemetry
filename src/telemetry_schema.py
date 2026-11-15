"""Central telemetry schema for the AI Analysis pipeline.

One repository-held file (``config/telemetry_schema.json``) declares, for every column the
pipeline supports, what it physically is, its category, its role, its unit and whether that
unit is verified. Nothing else in the pipeline guesses those facts.

Rules
-----
* A unit is **applied** only when ``unit_status == "verified"``. Every other level is shown as
  a hypothesis at most; the value stays in the file's own raw numbers.
* Recognition uses the header set (case/underscore/space-insensitive) plus declared aliases,
  never the filename.
* Missing optional columns, extra unknown columns and different column order are tolerated.
  An unknown column is returned as unmapped; its meaning is never invented.
* Axis/sign conventions are not part of this file. They are not established for this dataset,
  so no direction of motion (left/right, up/down) is derived from signs.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA = ROOT / "config" / "telemetry_schema.json"

UNIT_STATUSES = ("verified", "supported", "unverified", "unknown")
ROLE_MEASUREMENT = "measurement"


class SchemaError(ValueError):
    """The schema file is malformed."""


def _n(name: str) -> str:
    return re.sub(r"[\s_\-.]+", "", str(name)).lower()


@dataclass(frozen=True)
class ColumnSpec:
    name: str                      # column name as declared
    canonical: str                 # pipeline signal name
    category: str                  # attitude | acceleration | angular_rate | altitude | distance | motor | command | status | control | aux
    role: str                      # measurement | command | output | status | internal
    unit: Optional[str] = None
    unit_status: str = "unknown"
    description: str = ""
    basis: str = ""
    physical_range: Optional[Tuple[float, float]] = None
    min_excitation: Optional[float] = None   # smallest command change that can excite a response (raw units)
    aliases: Tuple[str, ...] = ()

    @property
    def unit_applied(self) -> bool:
        return bool(self.unit) and self.unit_status == "verified"

    @property
    def display_unit(self) -> Optional[str]:
        return self.unit if self.unit_applied else None

    def public(self) -> Dict[str, Any]:
        return {
            "name": self.name, "canonical": self.canonical, "category": self.category, "role": self.role,
            "unit": self.display_unit, "unit_status": self.unit_status, "description": self.description,
            "basis": self.basis, "physical_range": list(self.physical_range) if self.physical_range else None,
            "min_excitation": self.min_excitation,
        }


@dataclass(frozen=True)
class TimeSpec:
    column: str
    unit: Optional[str]
    unit_status: str
    basis: str = ""

    @property
    def scale_to_s(self) -> Optional[float]:
        return {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}.get(self.unit or "") if self.unit_status == "verified" else None


@dataclass(frozen=True)
class RelationshipSpec:
    id: str
    label: str
    command: Optional[str] = None
    response: Tuple[str, ...] = ()
    rate: Tuple[str, ...] = ()
    attitude: Tuple[str, ...] = ()


@dataclass(frozen=True)
class DatasetSchema:
    id: str
    label: str
    description: str
    documentation_status: str
    required: Tuple[str, ...]
    time: TimeSpec
    columns: Dict[str, ColumnSpec]            # canonical -> spec
    relationships: Tuple[RelationshipSpec, ...] = ()
    aliases: Dict[str, Tuple[str, ...]] = field(default_factory=dict)

    def by_canonical(self, canonical: str) -> Optional[ColumnSpec]:
        return self.columns.get(canonical)

    @property
    def label_version(self) -> str:
        return self.label


@dataclass
class SchemaMatch:
    schema: Optional[DatasetSchema]
    kind: str                                   # exact | compatible | partial | none
    mapping: Dict[str, ColumnSpec] = field(default_factory=dict)   # actual header -> spec
    unmapped: List[str] = field(default_factory=list)
    missing_required: List[str] = field(default_factory=list)

    @property
    def applies(self) -> bool:
        return self.kind in ("exact", "compatible") and self.schema is not None


def _validate_time(raw: Dict[str, Any], ds_id: str) -> TimeSpec:
    t = raw.get("time") or {}
    col = t.get("column")
    if not col:
        raise SchemaError(f"{ds_id}: time.column is required")
    status = t.get("unit_status", "unknown")
    if status not in UNIT_STATUSES:
        raise SchemaError(f"{ds_id}: time.unit_status must be one of {UNIT_STATUSES}")
    unit = t.get("unit")
    if status == "verified" and unit not in ("s", "ms", "us", "ns"):
        raise SchemaError(f"{ds_id}: a verified time unit must be one of s, ms, us, ns")
    return TimeSpec(column=str(col), unit=unit, unit_status=status, basis=str(t.get("basis", "")))


def _bounds(v: Any) -> Optional[Tuple[float, float]]:
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        return None
    try:
        lo, hi = float(v[0]), float(v[1])
    except (TypeError, ValueError):
        return None
    return (lo, hi) if lo < hi else None


def _validate_dataset(raw: Dict[str, Any]) -> DatasetSchema:
    ds_id = str(raw.get("id") or "unnamed")
    cols: Dict[str, ColumnSpec] = {}
    aliases: Dict[str, Tuple[str, ...]] = {str(k): tuple(str(a) for a in v or []) for k, v in (raw.get("aliases") or {}).items()}
    for name, c in (raw.get("columns") or {}).items():
        canon = c.get("canonical")
        if not canon:
            raise SchemaError(f"{ds_id}: column '{name}' has no canonical name")
        status = c.get("unit_status", "unknown")
        if status not in UNIT_STATUSES:
            raise SchemaError(f"{ds_id}: column '{name}' unit_status must be one of {UNIT_STATUSES}")
        unit = c.get("unit")
        if status == "verified" and not unit:
            raise SchemaError(f"{ds_id}: column '{name}' claims a verified unit but declares none")
        if status != "verified" and unit and c.get("unit") and status == "unknown":
            unit = None  # an unknown unit is not carried around as if it were known
        cols[str(canon)] = ColumnSpec(
            name=str(name), canonical=str(canon), category=str(c.get("category") or "unknown"),
            role=str(c.get("role") or ROLE_MEASUREMENT), unit=unit, unit_status=status,
            description=str(c.get("description") or ""), basis=str(c.get("basis") or ""),
            physical_range=_bounds(c.get("physical_range")),
            min_excitation=float(c["min_excitation"]) if c.get("min_excitation") is not None else None,
            aliases=tuple(aliases.get(str(name), ())),
        )
    rels: List[RelationshipSpec] = []
    for r in raw.get("relationships") or []:
        rels.append(RelationshipSpec(
            id=str(r["id"]), label=str(r.get("label") or r["id"]), command=r.get("command"),
            response=tuple(r.get("response") or ()), rate=tuple(r.get("rate") or ()),
            attitude=tuple(r.get("attitude") or ()),
        ))
    return DatasetSchema(
        id=ds_id, label=str(raw.get("label") or ds_id), description=str(raw.get("description") or ""),
        documentation_status=str(raw.get("documentation_status") or ""),
        required=tuple(str(s) for s in raw.get("required") or ()),
        time=_validate_time(raw, ds_id), columns=cols, relationships=tuple(rels), aliases=aliases,
    )


def load_schemas(path: Optional[Path | str] = None) -> List[DatasetSchema]:
    p = Path(path) if path else Path(os.environ.get("UAV_TELEMETRY_SCHEMA") or DEFAULT_SCHEMA)
    if not p.is_file():
        return []
    try:
        raw = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise SchemaError(f"{p}: {exc}") from exc
    out: List[DatasetSchema] = []
    for ds in raw.get("datasets", []):
        out.append(_validate_dataset(ds))
    return out


def _header_index(headers: Iterable[str]) -> Dict[str, str]:
    return {_n(h): h for h in headers}


def _lookup(names: Iterable[str], index: Dict[str, str]) -> Optional[str]:
    for n in names:
        if _n(n) in index:
            return index[_n(n)]
    return None


def match_schema(headers: List[str], schemas: Optional[List[DatasetSchema]] = None) -> SchemaMatch:
    """Match the header set to a declared dataset and resolve every recognised column."""
    index = _header_index(headers)
    best: Optional[SchemaMatch] = None
    for ds in schemas if schemas is not None else load_schemas():
        mapping: Dict[str, ColumnSpec] = {}
        for spec in ds.columns.values():
            names = [spec.name, spec.canonical, *spec.aliases, *ds.aliases.get(spec.name, ())]
            hit = _lookup(names, index)
            if hit is not None:
                mapping[hit] = spec
        missing = [r for r in ds.required if _lookup([r, *ds.aliases.get(r, ())], index) is None]
        used = set(mapping)
        time_hit = _lookup([ds.time.column, *ds.aliases.get(ds.time.column, ())], index)
        if time_hit is not None:
            used.add(time_hit)
        unmapped = [h for h in headers if h not in used]
        if time_hit is None:
            missing = missing + [ds.time.column]
        if not missing:
            exact = len(unmapped) == 0
            return SchemaMatch(ds, "exact" if exact else "compatible", mapping, unmapped, [])
        covered = len(used) / max(len(ds.columns) + 1, 1)
        if covered >= 0.7 and (best is None or covered > len(best.mapping) / max(len(ds.columns) + 1, 1)):
            best = SchemaMatch(none_schema(), "partial", mapping, unmapped, missing)
    return best or SchemaMatch(None, "none", {}, list(headers), [])


def none_schema() -> Optional[DatasetSchema]:
    return None


def resolve_time_header(headers: List[str], match: SchemaMatch) -> Optional[str]:
    """The actual header that carries time, including aliases."""
    if match.schema is None:
        return None
    index = _header_index(headers)
    return _lookup([match.schema.time.column, *match.schema.aliases.get(match.schema.time.column, ())], index)


def canonical_headers(headers: List[str], match: SchemaMatch) -> Dict[str, str]:
    """canonical signal -> actual header, for every resolved column."""
    return {spec.canonical: h for h, spec in match.mapping.items()}


def spec_for_canonical(match: SchemaMatch, canonical: str) -> Optional[ColumnSpec]:
    if match.schema is None:
        return None
    return match.schema.by_canonical(canonical)
