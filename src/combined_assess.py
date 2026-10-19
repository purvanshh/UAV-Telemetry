"""Compact, evidence-based assessment of a combined CSV against the deployed detector.

Every status here is computed from the actual mapping / validation results; nothing is
hard-coded text about a particular file.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.combined_ingest import (
    AMBIGUOUS,
    NEEDS_CONFIRMATION,
    REJECTED,
    TARGETS,
    UNIT_DATASET,
    UNIT_PROFILE,
    UNIT_USER,
    VALID,
    Inspection,
    SchemaMapping,
)
from src.dataset_profiles import Profile
from src.upload import UploadResult

# (key, label, canonical signals, role)
DETECTOR_INPUTS = [
    ("gyro", "Gyroscope X, Y, Z", ["gyro_x", "gyro_y", "gyro_z"], "scored"),
    ("accel", "Accelerometer X, Y, Z", ["acc_x", "acc_y", "acc_z"], "scored"),
    ("attitude", "Measured roll and pitch", ["roll", "pitch"], "scored"),
    ("yaw", "Yaw (heading)", ["yaw"], "scored"),
    ("desired", "Desired roll and pitch (used for tracking residuals)", ["des_roll", "des_pitch"], "scored"),
    ("baro", "Barometer: altitude, pressure, temperature", ["alt", "press", "temp"], "pipeline"),
]

UNIT_SOURCE_TEXT = {
    UNIT_DATASET: "defined by the dataset column name",
    UNIT_PROFILE: "stated by the dataset profile",
    UNIT_USER: "stated by you",
}

READY, NEEDS_INPUT, INCOMPATIBLE, INVALID = "ready", "needs_input", "incompatible", "invalid"


def _state(m: SchemaMapping, canon: str) -> str:
    cols = [c for c in m.columns if c.canonical == canon]
    if any(c.validation_status == VALID for c in cols):
        return "ok"
    if any(c.validation_status == AMBIGUOUS for c in cols):
        return "ambiguous"
    if any(c.validation_status == NEEDS_CONFIRMATION for c in cols):
        return "unit_unresolved"
    if any(c.validation_status == REJECTED for c in cols):
        return "rejected"
    return "missing"


def build_assessment(
    insp: Optional[Inspection],
    mapping: Optional[SchemaMapping],
    result: Optional[UploadResult],
    ts: Optional[Dict[str, Any]],
    profile: Optional[Profile],
    file_errors: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if insp is None or mapping is None:
        return {
            "outcome": INVALID, "compatible": False, "headline": "The file could not be read",
            "checks": [{"id": "csv", "status": "fail", "text": e} for e in (file_errors or ["Unreadable file."])],
            "detector_inputs": [], "missing": [], "unresolved": [], "needed": [], "counts": {},
            "profile": None,
        }
    m = mapping
    used = [c for c in m.columns if c.validation_status == VALID and c.canonical]
    ignored = [c for c in m.columns if c.ignored]
    unresolved_cols = [c for c in m.columns if c.canonical and c.validation_status in (NEEDS_CONFIRMATION, AMBIGUOUS, REJECTED)]
    unrecognised = [c for c in m.columns if not c.canonical and not c.ignored]
    required_targets = [t for t in TARGETS.values() if t.required]
    req_ok = [t for t in required_targets if _state(m, t.canonical) == "ok"]

    # --- per detector input group
    groups = []
    for key, label, canons, role in DETECTOR_INPUTS:
        states = {c: _state(m, c) for c in canons}
        if all(v == "ok" for v in states.values()):
            st = "ok"
        elif any(v == "missing" for v in states.values()):
            st = "missing"
        elif any(v == "ambiguous" for v in states.values()):
            st = "ambiguous"
        else:
            st = "unit_unresolved"
        src = {c.canonical: c.source_column for c in m.columns if c.canonical in canons and c.validation_status == VALID}
        groups.append({"key": key, "label": label, "role": role, "status": st, "signals": states, "sources": src})

    absent = [c for c in m.missing_required if _state(m, c) == "missing"]
    missing = [{"canonical": c, "label": TARGETS[c].label, "unit": TARGETS[c].unit,
                "group": TARGETS[c].group} for c in absent]
    unresolved = [
        {"column": c.source_column, "canonical": c.canonical, "label": TARGETS[c.canonical].label,
         "issue": c.unresolved or "meaning", "detail": c.ambiguity, "expected_unit": TARGETS[c.canonical].unit,
         "required": c.required}
        for c in unresolved_cols if c.required
    ]
    time_unresolved = m.time is not None and m.time.validation_status != VALID
    if time_unresolved:
        unresolved.insert(0, {"column": m.time.source_column, "canonical": "__time__", "label": "time column",
                              "issue": "unit", "detail": m.time.ambiguity, "expected_unit": None, "required": True})

    # --- outcome
    val_ok = bool(result and result.validation and result.validation.ok)
    ts_err = bool(ts and ts.get("errors"))
    if m.time is None or absent:
        outcome = INCOMPATIBLE
    elif unresolved or not m.scoreable:
        outcome = NEEDS_INPUT
    elif ts_err or not val_ok:
        outcome = INVALID
    else:
        outcome = READY

    # --- checks (all derived from results)
    checks: List[Dict[str, Any]] = [
        {"id": "csv", "status": "ok", "text": f"CSV recognised: {len(insp.df):,} rows, {len(insp.headers)} columns."}
    ]
    if profile:
        checks.append({"id": "profile", "status": "ok", "text": f"Known layout: {profile.id}."})
    if m.time is None:
        checks.append({"id": "timestamp", "status": "fail", "text": "No time column found."})
    elif m.time.validation_status != VALID:
        checks.append({"id": "timestamp", "status": "warn", "text": m.time.ambiguity or "Time unit not established."})
    elif ts_err:
        checks.append({"id": "timestamp", "status": "fail", "text": "; ".join(ts["errors"])})
    else:
        hz = f", about {ts['est_hz']:.1f} Hz" if ts and ts.get("est_hz") else ""
        checks.append({"id": "timestamp", "status": "ok",
                       "text": f"Timestamp interpretation validated: '{m.time.source_column}' in {m.time.unit} "
                               f"({UNIT_SOURCE_TEXT.get(m.time.unit_status, m.time.unit_status)}){hz}."})
    checks.append({
        "id": "mapping", "status": "ok" if len(req_ok) == len(required_targets) else ("fail" if absent else "warn"),
        "text": f"{len(req_ok)} of {len(required_targets)} required signals mapped.",
    })
    converted = [c for c in used if (c.scale, c.offset) != (1.0, 0.0)]
    if unresolved_cols and any(c.required for c in unresolved_cols):
        checks.append({"id": "units", "status": "warn",
                       "text": f"{sum(1 for c in unresolved_cols if c.required)} mapped signal(s) have no established unit or meaning."})
    elif used:
        checks.append({"id": "units", "status": "ok",
                       "text": f"Units established for all {len(used)} mapped signals"
                               + (f"; {len(converted)} converted exactly to canonical units." if converted else ".")})
    if result and result.validation is not None:
        checks.append({"id": "quality", "status": "ok" if val_ok else "fail",
                       "text": "Data quality checks passed." if val_ok else "Data quality checks failed: " + "; ".join(result.validation.errors[:3])})
    else:
        checks.append({"id": "quality", "status": "skip", "text": "Data quality checks were not run because the mapping is incomplete."})

    # --- what is needed
    needed: List[str] = []
    for x in missing:
        if x["canonical"] == "yaw":
            needed.append("A recorded yaw (heading) column in degrees. Yaw cannot be derived from other columns.")
        elif x["canonical"] in ("des_roll", "des_pitch"):
            needed.append(f"A column with the aircraft's desired (setpoint) {x['canonical'][4:]} in degrees. Pilot stick inputs do not qualify.")
        else:
            needed.append(f"A column for {x['label']} ({x['unit']}).")
    for u in unresolved:
        needed.append(f"{u['column']}: {u['detail']}")

    headline = {
        READY: "Ready to analyse",
        NEEDS_INPUT: "One more fact is needed before analysis",
        INCOMPATIBLE: "The deployed detector cannot analyse this file",
        INVALID: "The data did not pass validation",
    }[outcome]
    if outcome == INVALID:
        needed = list(result.all_errors[:5]) if result else list(ts.get("errors", []) if ts else [])
    return {
        "outcome": outcome,
        "compatible": outcome == READY,
        "headline": headline,
        "checks": checks,
        "detector_inputs": groups,
        "missing": missing,
        "unresolved": unresolved,
        "needed": needed,
        "counts": {
            "columns": len(insp.headers), "used": len(used), "converted": len(converted),
            "ignored": len(ignored), "unrecognised": len(unrecognised), "unresolved": len(unresolved),
            "required_total": len(required_targets), "required_mapped": len(req_ok),
        },
        "profile": {"id": profile.id, "documentation_status": profile.documentation_status} if profile else None,
        "pipeline_note": "Barometer columns are needed by the current pipeline for time alignment but are not "
                         "scored by the detector." if any(g["key"] == "baro" and g["status"] != "ok" for g in groups) else None,
    }
