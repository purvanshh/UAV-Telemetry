"""Dataset profiles: explicit, repository-held descriptions of known CSV layouts.

Units are never inferred from header text or value magnitudes. A unit is *established* only by
(1) the built-in dataset column names (``src/schema.py``) or (2) a profile in
``config/dataset_profiles.json`` (override the path with ``UAV_PROFILES_FILE``). A profile column
with ``"unit": null`` is deliberately unresolved and is reported as such.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILES = ROOT / "config" / "dataset_profiles.json"


class ProfileError(ValueError):
    pass


def _n(s: str) -> str:
    return re.sub(r"[\s_]+", "", str(s)).lower()


@dataclass(frozen=True)
class ProfileColumn:
    canonical: str
    unit: Optional[str]
    unresolved: Optional[str] = None


@dataclass(frozen=True)
class Profile:
    id: str
    description: str
    documentation_status: str
    signature: List[str]
    time_column: Optional[str]
    time_unit: Optional[str]
    columns: Dict[str, ProfileColumn]
    ignored: List[str] = field(default_factory=list)
    ignored_patterns: List[str] = field(default_factory=list)

    def matches(self, headers: List[str]) -> bool:
        have = {_n(h) for h in headers}
        return bool(self.signature) and all(_n(s) in have for s in self.signature)

    def column_for(self, header: str) -> Optional[ProfileColumn]:
        for k, v in self.columns.items():
            if _n(k) == _n(header):
                return v
        return None

    def is_ignored(self, header: str) -> bool:
        if any(_n(i) == _n(header) for i in self.ignored):
            return True
        return any(re.search(p, header, re.I) for p in self.ignored_patterns)


def _validate(raw: Dict[str, Any]) -> Profile:
    from src.combined_ingest import TARGETS, TIME_SCALE_TO_US, conversion, normalize_unit

    try:
        pid = str(raw["id"])
        cols: Dict[str, ProfileColumn] = {}
        for h, c in (raw.get("columns") or {}).items():
            canon = c["canonical"]
            if canon not in TARGETS:
                raise ProfileError(f"profile {pid}: '{h}' maps to unknown signal '{canon}'")
            unit = normalize_unit(c.get("unit")) if c.get("unit") is not None else None
            if c.get("unit") is not None and (unit is None or conversion(unit, TARGETS[canon].unit) is None):
                raise ProfileError(
                    f"profile {pid}: unit '{c.get('unit')}' of '{h}' cannot be converted exactly to "
                    f"{TARGETS[canon].unit}"
                )
            cols[h] = ProfileColumn(canon, unit, c.get("unresolved"))
        t = raw.get("time") or {}
        tu = normalize_unit(t.get("unit")) if t.get("unit") else None
        if t and tu not in TIME_SCALE_TO_US:
            raise ProfileError(f"profile {pid}: time unit must be one of s, ms, us, ns")
        return Profile(
            id=pid, description=str(raw.get("description", "")),
            documentation_status=str(raw.get("documentation_status", "")),
            signature=[str(s) for s in raw.get("signature", [])],
            time_column=t.get("column"), time_unit=tu, columns=cols,
            ignored=[str(s) for s in raw.get("ignored", [])],
            ignored_patterns=[str(s) for s in raw.get("ignored_patterns", [])],
        )
    except KeyError as exc:
        raise ProfileError(f"profile is missing {exc}") from exc


def load_profiles(path: Optional[Path | str] = None) -> List[Profile]:
    p = Path(path) if path else Path(os.environ.get("UAV_PROFILES_FILE") or DEFAULT_PROFILES)
    if not p.is_file():
        return []
    try:
        data = json.loads(p.read_text())
        return [_validate(x) for x in data.get("profiles", [])]
    except (json.JSONDecodeError, ProfileError) as exc:
        raise ProfileError(f"{p}: {exc}") from exc


def match_profile(headers: List[str], profiles: Optional[List[Profile]] = None) -> Optional[Profile]:
    for pr in (profiles if profiles is not None else load_profiles()):
        if pr.matches(headers):
            return pr
    return None
