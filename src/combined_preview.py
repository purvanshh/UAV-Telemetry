"""Orchestration for combined CSVs: inspect → map → (optional AI) → validate → assess.

Used by the preview endpoint and by the one-step automatic upload. Stores nothing itself.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from src.combined_assess import build_assessment
from src.combined_ingest import (
    AMBIGUOUS,
    NEEDS_CONFIRMATION,
    REJECTED,
    TARGETS,
    UNMAPPED,
    VALID,
    CombinedFileError,
    Inspection,
    SchemaMapping,
    check_timestamps,
    inspect_combined,
    load_combined,
    propose_mapping,
)
from src.dataset_profiles import Profile, ProfileMatch, match_profile_detailed
from src.schema_llm import LLM_DISCLOSURE, LLMSchemaResult, propose_schema
from src.upload import UploadResult

MAX_ASSIGNMENTS = 200

def _accepted_units(t) -> List[str]:
    from src.combined_ingest import _CONVERSIONS

    return sorted({t.unit} | {u for (u, v) in _CONVERSIONS if v == t.unit})


TARGET_OPTIONS = [
    {"canonical": t.canonical, "group": t.group, "label": t.label, "expected_unit": t.unit, "required": t.required,
     "accepted_units": _accepted_units(t)}
    for t in TARGETS.values()
]


def parse_assignments(raw: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """Parse the client's corrections: ``{"col": {"canonical"?: ..., "unit": ...}}``."""
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"assignments must be valid JSON: {exc}") from exc
    if not isinstance(data, dict) or len(data) > MAX_ASSIGNMENTS:
        raise ValueError("assignments must be an object of at most 200 columns.")
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in data.items():
        if not isinstance(k, str) or not isinstance(v, dict):
            raise ValueError("each assignment must be {canonical, unit}.")
        out[k] = {"canonical": v.get("canonical"), "unit": v.get("unit")}
    return out


def _column_status(cm) -> str:
    if cm.ignored:
        return "ignored"
    return {VALID: "recognized", NEEDS_CONFIRMATION: "needs_confirmation", AMBIGUOUS: "ambiguous",
            REJECTED: "rejected"}.get(cm.validation_status, "unknown")


class Analysis:
    """Everything derived from one pass over a combined file."""

    def __init__(self, insp, mapping, result, ts, llm, profile, assessment, errors, match=None):
        self.insp: Optional[Inspection] = insp
        self.mapping: Optional[SchemaMapping] = mapping
        self.result: Optional[UploadResult] = result
        self.ts = ts
        self.llm: LLMSchemaResult = llm
        self.profile: Optional[Profile] = profile
        self.assessment: Dict[str, Any] = assessment
        self.errors: List[str] = errors
        self.match: Optional[ProfileMatch] = match


def analyze_combined(
    raw: bytes,
    filename: str,
    assignments: Dict[str, Dict[str, Any]],
    llm_consent: bool,
    *,
    llm_client: Any = None,
    profile: Optional[Profile] = None,
) -> Analysis:
    try:
        insp = inspect_combined(raw, filename)
    except CombinedFileError as exc:
        empty = LLMSchemaResult(status="not_requested", message="")
        return Analysis(None, None, None, None, empty, None, build_assessment(None, None, None, None, None, [str(exc)]), [str(exc)])

    match = match_profile_detailed(insp.headers) if profile is None else ProfileMatch(profile, "compatible", "")
    profile = profile or match.profile
    mapping = propose_mapping(insp, assignments=assignments, profile=profile)
    # Only genuinely unfamiliar columns are ever sent: not recognised, not profile-ignored.
    unresolved = [
        c.source_column for c in mapping.columns
        if c.canonical is None and not c.ignored and c.validation_status == UNMAPPED and not (assignments.get(c.source_column) or {}).get("canonical")
    ]
    llm = propose_schema(insp, unresolved, consent=llm_consent, client=llm_client)
    if llm.proposals:
        mapping = propose_mapping(insp, assignments=assignments, llm_proposals=llm.proposals, profile=profile)

    ts = check_timestamps(insp, mapping.time if mapping.time and mapping.time.unit else None)
    result: Optional[UploadResult] = None
    errors = list(mapping.blockers)
    if mapping.scoreable:
        result, _i, _m = load_combined(raw, filename, assignments=assignments, llm_proposals=llm.proposals, profile=profile)
        errors = list(result.all_errors)
    return Analysis(insp, mapping, result, ts, llm, profile,
                    build_assessment(insp, mapping, result, ts, profile, match=match), errors, match)


def build_combined_preview(
    raw: bytes, filename: str, assignments: Dict[str, Dict[str, Any]], llm_consent: bool, *, llm_client: Any = None,
    profile: Optional[Profile] = None,
) -> Dict[str, Any]:
    a = analyze_combined(raw, filename, assignments, llm_consent, llm_client=llm_client, profile=profile)
    return preview_from_analysis(a, filename)


def preview_from_analysis(a: Analysis, filename: str) -> Dict[str, Any]:
    if a.insp is None or a.mapping is None:
        return {"ok": False, "filename": filename, "errors": a.errors, "scoreable": False, "assessment": a.assessment}
    insp, mapping = a.insp, a.mapping

    cols: List[Dict[str, Any]] = []
    if mapping.time:
        cols.append({"name": mapping.time.source_column, "status": _column_status(mapping.time), "role": "time",
                     **{k: v for k, v in mapping.time.to_dict().items() if k != "source_column"},
                     "samples": insp.stats[mapping.time.source_column]["samples"]})
    for c in mapping.columns:
        d = c.to_dict()
        d.pop("source_column")
        cols.append({"name": c.source_column, "status": _column_status(c),
                     "role": "signal" if c.canonical else ("ignored" if c.ignored else "unmapped"),
                     **d, "samples": insp.stats[c.source_column]["samples"],
                     "numeric_fraction": insp.stats[c.source_column]["numeric_fraction"]})
    counts = {k: sum(1 for c in cols if c["status"] == k)
              for k in ("recognized", "needs_confirmation", "ambiguous", "rejected", "unknown", "ignored")}
    validation = a.result.validation.to_dict() if a.result and a.result.validation else None
    scoreable = bool(a.assessment["compatible"])
    return {
        "ok": True, "filename": filename, "size_bytes": insp.size_bytes, "rows": int(len(insp.df)),
        "columns": cols, "counts": counts, "sample_rows": insp.preview_rows(5),
        "mapping": mapping.to_dict(), "timestamp": a.ts, "validation": validation,
        "scoreable": scoreable, "blockers": a.errors if not scoreable else [],
        "warnings": list(mapping.warnings) + (validation or {}).get("warnings", []),
        "targets": TARGET_OPTIONS, "assessment": a.assessment,
        "llm": {**a.llm.to_dict(), "disclosure": LLM_DISCLOSURE},
        "scoreable_note": (
            "This file maps completely onto the deployed detector's inputs and can be analysed."
            if scoreable else
            "This file cannot be analysed by the deployed detector as it stands. Nothing has been stored."
        ),
    }
