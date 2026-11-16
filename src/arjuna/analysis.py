"""One ARJUNA flight -> one deterministic analysis: overview, timeline, rule findings and physics findings.

This is everything the code knows before any language model is involved. The agent reads it through tools and may
investigate further, but it never changes these findings.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from src.arjuna.checks import FAULT_CODES, SEV_RANK, _r, clock, run_checks, worst_severity
from src.arjuna.physics import run_physics
from src.arjuna.ulog import ArjunaFlight, enum_name

ANALYSIS_VERSION = "arjuna-1"

# Every topic in the standard (v1.1 topic list), so missing ones can be named.
STANDARD_TOPICS = [
    "arjuna_vehicle_status", "arjuna_imu", "arjuna_imu_filtered", "arjuna_vibration", "arjuna_baro", "arjuna_gps",
    "arjuna_mag", "arjuna_rangefinder", "arjuna_optical_flow", "arjuna_vision_pose", "arjuna_attitude", "arjuna_position",
    "arjuna_estimator_innov", "arjuna_setpoints", "arjuna_rc_input", "arjuna_controller", "arjuna_actuator_outputs",
    "arjuna_esc_status", "arjuna_battery", "arjuna_fpga_timing", "arjuna_system_health", "arjuna_flight_event",
    "arjuna_mission_status", "arjuna_communication", "arjuna_firewall", "arjuna_fallback", "arjuna_proximity",
    "arjuna_detection", "arjuna_log_hash",
]
IDENTITY_KEYS = ["flight_id", "vehicle_id", "board_serial", "hw_version", "soc", "fw_version", "fw_build_id",
                 "bitstream_version", "schema_version", "log_profile", "sensors_installed", "boot_count", "source_format",
                 "source_file", "source_profile", "converter_version", "enum_table_version", "motor_layout"]


def _segments(t: np.ndarray, v: np.ndarray, table: str, limit: int = 40) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not len(t):
        return out
    start = 0
    for i in range(1, len(v) + 1):
        if i == len(v) or (v[i] != v[start] and not (np.isnan(v[i]) and np.isnan(v[start]))):
            name = enum_name(table, v[start]) if np.isfinite(v[start]) else "unknown"
            out.append({"value": name, "t_start": _r(t[start], 2), "t_end": _r(t[i - 1], 2),
                        "when": f"{clock(t[start])}–{clock(t[i - 1])}"})
            start = i
    return out[:limit]


def timeline(fl: ArjunaFlight) -> Dict[str, Any]:
    out: Dict[str, Any] = {"duration_s": _r(fl.duration_s, 1), "airborne_s": _r(fl.airborne_s(), 1),
                           "air_intervals": [{"t_start": _r(a, 2), "t_end": _r(b, 2), "when": f"{clock(a)}–{clock(b)}"}
                                             for a, b in fl.air_intervals()]}
    if out["air_intervals"]:
        out["takeoff_t"] = out["air_intervals"][0]["t_start"]
        out["landing_t"] = out["air_intervals"][-1]["t_end"]
    vs = fl.get("arjuna_vehicle_status")
    if vs is not None:
        t = vs["t"].to_numpy(float)
        out["modes"] = _segments(t, vs["flight_mode"].to_numpy(float), "FLIGHT_MODE") if "flight_mode" in vs else []
        if "armed" in vs:
            out["armed"] = [s for s in _segments(t, vs["armed"].to_numpy(float), "_") if s["value"] == "_1"]
            for s in out["armed"]:
                s["value"] = "armed"
        if "control_source" in vs:
            out["control_source"] = _segments(t, vs["control_source"].to_numpy(float), "CONTROL_SOURCE")
    fe = fl.get("arjuna_flight_event")
    events = []
    if fe is not None:
        for _, row in fe.iterrows():
            if not np.isfinite(row.get("event_code", np.nan)):
                continue
            events.append({"t": _r(row["t"], 2), "at": clock(row["t"]), "event": enum_name("EVENT_CODES", row["event_code"]),
                           "severity": enum_name("SEVERITY", row.get("severity", np.nan)),
                           "category": enum_name("EVENT_CATEGORY", row.get("category", np.nan)),
                           **({"mode_before": enum_name("FLIGHT_MODE", row["mode_before"]),
                               "mode_after": enum_name("FLIGHT_MODE", row["mode_after"])}
                              if row.get("mode_before") != row.get("mode_after") and np.isfinite(row.get("mode_after", np.nan)) else {})})
    out["events"] = events[:200]
    out["dropouts"] = fl.dropouts[:50]
    levels: Dict[str, int] = {}
    for m in fl.messages:
        levels[m["level"]] = levels.get(m["level"], 0) + 1
    out["message_counts"] = levels
    out["important_messages"] = [{**m, "at": clock(m["t"])} for m in fl.messages
                                 if m["level"] in ("emergency", "alert", "critical", "error", "warning")][:60]
    return out


def topic_table(fl: ArjunaFlight) -> Dict[str, Any]:
    rows = []
    for (name, inst), tp in sorted(fl.topics.items()):
        rows.append({"topic": name, "instance": inst, "samples": int(len(tp.df)), "rate_hz": tp.rate_hz,
                     "fields": tp.fields, "unavailable_fields": tp.unavailable})
    present = {r["topic"] for r in rows}
    not_produced = [t.strip() for t in str(fl.info.get("topics_not_produced", "")).split(",") if t.strip()]
    return {"topics": rows, "missing_standard_topics": [t for t in STANDARD_TOPICS if t not in present],
            "topics_not_produced": not_produced}


def analyse(fl: ArjunaFlight) -> Dict[str, Any]:
    if not fl.compatibility["ok"]:
        return {"analysis_version": ANALYSIS_VERSION, "kind": "ulog", "refused": True,
                "identity": {k: fl.info.get(k) for k in IDENTITY_KEYS if k in fl.info},
                "compatibility": fl.compatibility, "integrity": fl.integrity, "findings": [], "physics": {},
                "file": {"name": fl.filename, "sha256": fl.sha256, "size_bytes": fl.size_bytes}}
    rules = run_checks(fl)
    phys = run_physics(fl) if rules["classification"]["kind"] == "flight" else {
        "hover_balance": {"status": "not_computable", "reason": "not a flight"}, "thrust_residual": {"status": "not_computable"},
        "dynamics": {"status": "not_computable"}, "findings": []}
    findings = rules["findings"] + phys["findings"]
    if phys["findings"]:
        rules["coverage"]["MOTOR_ESC"]["status"] = "evaluated"
        rules["coverage"]["MOTOR_ESC"]["reasons"] = []
    for f in phys["findings"]:
        if f["detector"] not in rules["coverage"][f["code"]]["checks"]:
            rules["coverage"][f["code"]]["checks"].append(f["detector"])
    if phys["hover_balance"].get("status") == "computed":
        rules["coverage"]["MOTOR_ESC"]["status"] = "evaluated"
        rules["coverage"]["MOTOR_ESC"]["reasons"] = []
        if "hover motor balance" not in rules["coverage"]["MOTOR_ESC"]["checks"]:
            rules["coverage"]["MOTOR_ESC"]["checks"].append("hover motor balance")
    elif rules["coverage"]["MOTOR_ESC"]["status"] != "evaluated":
        rules["coverage"]["MOTOR_ESC"]["reasons"].append(phys["hover_balance"].get("reason", "hover balance not computable"))
    findings.sort(key=lambda f: (SEV_RANK.get(f["severity"], 3), f["t_start"] if f["t_start"] is not None else 1e12))
    for i, f in enumerate(findings, 1):
        f["id"] = f"F{i}"
    sev = worst_severity(findings)
    return {
        "analysis_version": ANALYSIS_VERSION, "kind": "ulog", "refused": False,
        "file": {"name": fl.filename, "sha256": fl.sha256, "size_bytes": fl.size_bytes, "folder_files": fl.folder_files},
        "identity": {k: fl.info.get(k) for k in IDENTITY_KEYS if k in fl.info},
        "compatibility": fl.compatibility, "integrity": fl.integrity, "classification": rules["classification"],
        "timeline": timeline(fl), "topics": topic_table(fl), "findings": findings, "coverage": rules["coverage"],
        "metrics": rules["metrics"], "physics": {k: v for k, v in phys.items() if k != "findings"},
        "thresholds": rules["thresholds"], "fault_codes": FAULT_CODES,
        "verdict": {"worst": sev, "critical": sum(f["severity"] == "critical" for f in findings),
                    "warning": sum(f["severity"] == "warning" for f in findings),
                    "codes": sorted({f["code"] for f in findings})},
        "metadata": fl.metadata, "engineer_labels": _labels(fl),
    }


def _labels(fl: ArjunaFlight) -> Optional[Dict[str, Any]]:
    """labels.json from the flight folder, for comparison only. It is never shown to the agent."""
    if not fl.labels:
        return None
    labs = []
    for lab in fl.labels.get("labels", []) or []:
        try:
            s, e = (int(lab["start_us"]) - fl.t0_us) / 1e6, (int(lab["end_us"]) - fl.t0_us) / 1e6
        except (KeyError, TypeError, ValueError):
            continue
        labs.append({"code": lab.get("code"), "severity": lab.get("severity"), "t_start": _r(s, 2), "t_end": _r(e, 2),
                     "when": f"{clock(s)}–{clock(e)}", "note": lab.get("note"), "source": lab.get("source")})
    return {"reviewed": fl.labels.get("reviewed"), "reviewed_by": fl.labels.get("reviewed_by"),
            "method": fl.labels.get("review_method"), "split": fl.labels.get("split"), "labels": labs}
