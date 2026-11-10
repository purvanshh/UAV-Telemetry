"""Dataset profiles: explicit, versioned, repository-held descriptions of known CSV layouts.

A profile records, per column: its role, canonical signal (if any), unit, and — separately —
how well each fact is established (``verified`` vs ``unverified``) with the evidence for it.
Only *verified* units and frames let a mapping become ``valid``. Hypotheses are kept and shown,
never silently promoted. Edit ``config/dataset_profiles.json`` (or point ``UAV_PROFILES_FILE``
elsewhere) when authoritative documentation establishes a fact.

Recognition never uses the filename: a file matches when all of the profile's *required* columns
are present (extra columns and missing optional ones are tolerated and reported). A file that has
most of a profile's columns but lacks required ones is reported as a ``partial`` match and the
profile is NOT applied.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILES = ROOT / "config" / "dataset_profiles.json"
DEFAULT_CONFIRMATIONS = ROOT / "config" / "profile_confirmations.json"

# Evidence levels for a unit or axis convention. Only ``verified`` is ever applied.
#   verified   - documentation or an explicit owner confirmation (recorded in the confirmations overlay)
#   supported  - physically cross-checked against the data (see docs/FLT_DATASET_INVESTIGATION.md);
#                shown to the operator, NOT applied until the owner confirms it
#   unverified - hypothesis only
LEVELS = ("verified", "supported", "unverified")
PARTIAL_MATCH_FRACTION = 0.7


class ProfileError(ValueError):
    pass


def _n(s: str) -> str:
    return re.sub(r"[\s_]+", "", str(s)).lower()


def headers_fingerprint(headers: List[str]) -> str:
    """Order-, case- and underscore-insensitive hash of a header set (no filename, no row count)."""
    return hashlib.sha256("|".join(sorted(_n(h) for h in headers)).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class ProfileColumn:
    canonical: Optional[str]  # pipeline signal, or None for context-only columns
    unit: Optional[str] = None  # unit used for conversion (hypothesis if unit_status != verified)
    unresolved: Optional[str] = None  # human-readable open question
    role: str = "signal"  # signal | status | command | motor | control | distance | vertical
    unit_status: str = "verified"  # verified | unverified
    frame_status: str = "verified"  # axis/sign convention: verified | supported | unverified
    frame_sign: int = 1  # sign to multiply by to reach the model's convention (applied only when verified)
    frame_expr: Optional[str] = None  # signed choice-fact reference, e.g. "-pitch_sign": sign DERIVED from confirmed facts
    fact: Optional[str] = None  # confirmation key for this column's unit (see FACTS)
    frame_fact: Optional[str] = None  # confirmation key for this column's frame
    status: str = "verified"  # semantic status of the column meaning: verified | candidate | unverified
    evidence: str = ""
    handling: str = ""
    never_maps_to: List[str] = field(default_factory=list)
    unit_hypothesis: Optional[str] = None  # shown in the question, never applied


@dataclass(frozen=True)
class Profile:
    id: str
    version: str
    description: str
    documentation_status: str
    required: List[str]
    time_column: Optional[str]
    time_unit: Optional[str]
    time_unit_status: str
    time_evidence: str
    columns: Dict[str, ProfileColumn]
    baro_required: bool = True  # False: barometer columns are optional for this layout
    choices: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # owner-selectable convention facts
    applied_confirmations: List[Dict[str, Any]] = field(default_factory=list)
    confirmation_notes: List[str] = field(default_factory=list)
    ignored: List[str] = field(default_factory=list)
    ignored_patterns: List[str] = field(default_factory=list)
    missing_notes: Dict[str, str] = field(default_factory=dict)  # canonical -> why it can't be supplied
    raw_fingerprint: Optional[str] = None

    @property
    def signature(self) -> List[str]:  # backwards-compatible name
        return self.required

    @property
    def registered_headers(self) -> List[str]:
        hs = list(self.columns) + list(self.ignored)
        if self.time_column:
            hs.append(self.time_column)
        return hs

    @property
    def fingerprint(self) -> str:
        return headers_fingerprint(self.registered_headers)

    @property
    def label(self) -> str:
        return f"{self.id} v{self.version}"

    def matches(self, headers: List[str]) -> bool:
        have = {_n(h) for h in headers}
        return bool(self.required) and all(_n(s) in have for s in self.required)

    def column_for(self, header: str) -> Optional[ProfileColumn]:
        for k, v in self.columns.items():
            if _n(k) == _n(header):
                return v
        return None

    def is_ignored(self, header: str) -> bool:
        if any(_n(i) == _n(header) for i in self.ignored):
            return True
        return any(re.search(p, header, re.I) for p in self.ignored_patterns)


@dataclass
class ProfileMatch:
    profile: Optional[Profile]
    kind: str  # exact | compatible | partial | none
    fingerprint: str
    missing_required: List[str] = field(default_factory=list)
    missing_optional: List[str] = field(default_factory=list)
    extra_columns: List[str] = field(default_factory=list)
    candidate: Optional[Profile] = None  # the near-miss profile when kind == partial

    @property
    def applies(self) -> bool:
        return self.kind in ("exact", "compatible")

    def to_dict(self) -> Dict[str, Any]:
        p = self.profile or self.candidate
        return {
            "kind": self.kind, "profile": p.label if p else None, "profile_id": p.id if p else None,
            "version": p.version if p else None, "file_fingerprint": self.fingerprint,
            "profile_fingerprint": p.fingerprint if p else None, "missing_required": self.missing_required,
            "missing_optional": self.missing_optional, "extra_columns": self.extra_columns,
        }


def _validate(raw: Dict[str, Any]) -> Profile:
    from src.combined_ingest import TARGETS, TIME_SCALE_TO_US, conversion, normalize_unit

    try:
        pid = str(raw["id"])
        cols: Dict[str, ProfileColumn] = {}
        for h, c in (raw.get("columns") or {}).items():
            canon = c.get("canonical")
            if canon is not None and canon not in TARGETS:
                raise ProfileError(f"profile {pid}: '{h}' maps to unknown signal '{canon}'")
            unit = normalize_unit(c.get("unit")) if c.get("unit") is not None else None
            if c.get("unit") is not None and (
                unit is None or canon is None or conversion(unit, TARGETS[canon].unit) is None
            ):
                raise ProfileError(f"profile {pid}: unit '{c.get('unit')}' of '{h}' cannot be converted exactly")
            for st_key in ("unit_status", "frame_status"):
                if c.get(st_key, "verified") not in LEVELS:
                    raise ProfileError(f"profile {pid}: '{h}' {st_key} must be one of {LEVELS}")
            if c.get("frame_sign", 1) not in (1, -1):
                raise ProfileError(f"profile {pid}: '{h}' frame_sign must be 1 or -1")
            hyp = c.get("unit_hypothesis")
            cols[h] = ProfileColumn(
                canonical=canon, unit=unit, unresolved=c.get("unresolved"), role=c.get("role", "signal"),
                unit_status=c.get("unit_status", "verified"), frame_status=c.get("frame_status", "verified"),
                status=c.get("status", "verified"), evidence=c.get("evidence", ""), handling=c.get("handling", ""),
                never_maps_to=list(c.get("never_maps_to", [])), unit_hypothesis=hyp,
                frame_sign=int(c.get("frame_sign", 1)), fact=c.get("fact"), frame_fact=c.get("frame_fact"),
                frame_expr=c.get("frame_expr"),
            )
        t = raw.get("time") or {}
        tu = normalize_unit(t.get("unit")) if t.get("unit") else None
        if t and tu not in TIME_SCALE_TO_US:
            raise ProfileError(f"profile {pid}: time unit must be one of s, ms, us, ns")
        if t.get("unit_status", "verified") not in LEVELS:
            raise ProfileError(f"profile {pid}: time unit_status must be one of {LEVELS}")
        choices = (raw.get("frame_model") or {}).get("choices", {})
        for name, ch in choices.items():
            if not ch.get("values") or any(v not in (1, -1) for v in ch["values"]):
                raise ProfileError(f"profile {pid}: choice '{name}' must list values from [1, -1]")
        for h, c in cols.items():
            if c.frame_expr:
                ref = c.frame_expr.lstrip("-")
                if ref not in choices:
                    raise ProfileError(f"profile {pid}: '{h}' frame_expr references unknown choice '{ref}'")
        required = [str(s) for s in raw.get("required", raw.get("signature", []))]
        for r in required:
            if not any(_n(r) == _n(k) for k in list(cols) + ([t["column"]] if t.get("column") else [])):
                raise ProfileError(f"profile {pid}: required column '{r}' is not described by the profile")
        return Profile(
            id=pid, version=str(raw.get("version", "0")), description=str(raw.get("description", "")),
            documentation_status=str(raw.get("documentation_status", "")), required=required,
            time_column=t.get("column"), time_unit=tu, time_unit_status=t.get("unit_status", "verified"),
            time_evidence=t.get("evidence", ""), columns=cols,
            ignored=[str(s) for s in raw.get("ignored", [])],
            ignored_patterns=[str(s) for s in raw.get("ignored_patterns", [])],
            missing_notes=dict((raw.get("detector_compatibility") or {}).get("missing_notes", {})),
            baro_required=bool((raw.get("detector_compatibility") or {}).get("baro_required", True)),
            choices=dict((raw.get("frame_model") or {}).get("choices", {})),
        )
    except KeyError as exc:
        raise ProfileError(f"profile is missing {exc}") from exc


def _confirmations_path() -> Path:
    return Path(os.environ.get("UAV_PROFILE_CONFIRMATIONS") or DEFAULT_CONFIRMATIONS)


def fact_digest(profile: Profile, fact: str) -> str:
    """Digest of exactly what a confirmation of ``fact`` is a statement about.

    It covers the fact's proposed value or allowed choices and, for every column involved, the signal
    it is mapped to, its hypothesised unit, its frame expression and its recorded evidence. If any of
    that is edited after a confirmation was recorded, the digest no longer matches and the
    confirmation is ignored rather than silently transferred to a different claim.
    """
    items: List[Dict[str, Any]] = []
    if fact == "time_unit" and profile.time_column:
        items.append({"name": profile.time_column, "unit": profile.time_unit, "evidence": profile.time_evidence})
    for name, c in sorted(profile.columns.items()):
        if c.fact == fact or (c.frame_expr and c.frame_expr.lstrip("-") == fact):
            items.append({"name": name, "canonical": c.canonical, "unit": c.unit, "unit_hypothesis": c.unit_hypothesis,
                          "frame_expr": c.frame_expr, "evidence": c.evidence})
    payload = {"profile": profile.id, "version": profile.version, "fact": fact,
               "allowed": (profile.choices.get(fact) or {}).get("values"), "meaning": (profile.choices.get(fact) or {}).get("meaning"),
               "items": items}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _signed_ref(expr: str) -> tuple[int, str]:
    return (-1, expr[1:]) if expr.startswith("-") else (1, expr)


def apply_confirmations(profile: Profile, confirmations: List[Dict[str, Any]]) -> Profile:
    """Replay the append-only confirmation log onto one profile and promote exactly what it supports.

    A record counts only if it names this profile id and version, carries the digest of the fact as
    currently defined, has an author and a basis, and states an allowed value (the proposed unit, or
    one of a choice fact's listed values). Later records supersede earlier ones for the same fact;
    a ``revoke`` removes it. Everything ignored is reported in ``confirmation_notes``.
    """
    from src.combined_ingest import normalize_unit

    notes: List[str] = []
    current: Dict[str, Dict[str, Any]] = {}
    for rec in confirmations:
        if rec.get("profile") != profile.id:
            continue  # belongs to another profile: never applied here
        fact = rec.get("fact")
        if str(rec.get("version")) != profile.version:
            notes.append(f"Ignored record for '{fact}': it was made for profile version {rec.get('version')}, this is {profile.version}.")
            continue
        if not rec.get("confirmed_by") or not str(rec.get("basis", "")).strip():
            notes.append(f"Ignored record for '{fact}': it lacks an author or a basis.")
            continue
        if rec.get("action", "confirm") == "revoke":
            current.pop(fact, None)
            continue
        if rec.get("fact_digest") != fact_digest(profile, fact):
            notes.append(f"Ignored confirmation of '{fact}': the fact's definition or evidence changed after it was recorded.")
            continue
        current[fact] = rec

    cols = dict(profile.columns)
    time_status, time_ev = profile.time_unit_status, profile.time_evidence
    applied: List[Dict[str, Any]] = []
    choice_value: Dict[str, int] = {}

    for fact, rec in current.items():
        value = str(rec.get("value"))
        stamp = f"owner confirmation by {rec.get('confirmed_by')} on {rec.get('confirmed_on', '?')}"
        ok = False
        if fact in profile.choices:
            if value in {str(v) for v in profile.choices[fact]["values"]}:
                choice_value[fact] = int(value)
                ok = True
        elif fact == "time_unit":
            if normalize_unit(value) == profile.time_unit:
                time_status, time_ev, ok = "verified", f"{profile.time_evidence} | {stamp}", True
        else:
            for name, col in list(cols.items()):
                if col.fact == fact:
                    want = col.unit or col.unit_hypothesis
                    if want and normalize_unit(value) == normalize_unit(want):
                        cols[name] = dataclasses.replace(col, unit=normalize_unit(want), unit_status="verified",
                                                         evidence=f"{col.evidence} | {stamp}")
                        ok = True
        if ok:
            applied.append({k: rec.get(k) for k in ("fact", "value", "confirmed_by", "confirmed_on", "recorded_at", "basis")})
        else:
            notes.append(f"Ignored confirmation of '{fact}' = '{value}': not an allowed value.")

    # frame signs are DERIVED from the confirmed choices; a column is verified only if every choice it uses is.
    for name, col in list(cols.items()):
        if col.frame_expr:
            sign, ref = _signed_ref(col.frame_expr)
            if ref in choice_value:
                cols[name] = dataclasses.replace(col, frame_status="verified", frame_sign=sign * choice_value[ref])
            else:
                cols[name] = dataclasses.replace(col, frame_status="unverified" if col.frame_status != "supported" else "supported")
    return dataclasses.replace(profile, columns=cols, time_unit_status=time_status, time_evidence=time_ev,
                               applied_confirmations=applied, confirmation_notes=notes)


def load_base_profiles(path: Optional[Path | str] = None) -> List[Profile]:
    """Profiles exactly as defined in the profile file (no owner confirmations applied)."""
    p = Path(path) if path else Path(os.environ.get("UAV_PROFILES_FILE") or DEFAULT_PROFILES)
    if not p.is_file():
        return []
    try:
        data = json.loads(p.read_text())
        return [_validate(x) for x in data.get("profiles", [])]
    except (json.JSONDecodeError, ProfileError) as exc:
        raise ProfileError(f"{p}: {exc}") from exc


def load_confirmation_log() -> List[Dict[str, Any]]:
    cp = _confirmations_path()
    if not cp.is_file():
        return []
    try:
        return list(json.loads(cp.read_text()).get("confirmations", []))
    except json.JSONDecodeError as exc:
        raise ProfileError(f"{cp}: {exc}") from exc


def load_profiles(path: Optional[Path | str] = None) -> List[Profile]:
    log = load_confirmation_log()
    return [apply_confirmations(pr, log) for pr in load_base_profiles(path)]


def match_profile_detailed(headers: List[str], profiles: Optional[List[Profile]] = None) -> ProfileMatch:
    fp = headers_fingerprint(headers)
    have = {_n(h): h for h in headers}
    best_partial: Optional[tuple] = None
    for pr in (profiles if profiles is not None else load_profiles()):
        reg = {_n(h) for h in pr.registered_headers}
        missing_req = [r for r in pr.required if _n(r) not in have]
        covered = len(reg & set(have)) / max(len(reg), 1)
        if not missing_req:
            extra = [have[k] for k in have if k not in reg]
            miss_opt = [h for h in pr.registered_headers if _n(h) not in have]
            kind = "exact" if fp == pr.fingerprint else "compatible"
            return ProfileMatch(pr, kind, fp, [], miss_opt, extra)
        if covered >= PARTIAL_MATCH_FRACTION and (best_partial is None or covered > best_partial[0]):
            best_partial = (covered, pr, missing_req)
    if best_partial:
        _c, pr, mr = best_partial
        return ProfileMatch(None, "partial", fp, mr, [], [], candidate=pr)
    return ProfileMatch(None, "none", fp)


def match_profile(headers: List[str], profiles: Optional[List[Profile]] = None) -> Optional[Profile]:
    return match_profile_detailed(headers, profiles).profile
