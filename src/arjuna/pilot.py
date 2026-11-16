"""The pilot's view of an analysis: one verdict, the problems in plain words, and what to check before the next flight.

Written by code from the findings, so it is the same with or without the AI agent. No topic names, no field names,
no jargon. Engineering detail stays in the findings themselves.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.arjuna.checks import clock

# Per fault code: plain name, what it means for the pilot, and what to check (warning / critical).
PLAIN: Dict[str, Dict[str, str]] = {
    "VIB_HIGH": {"name": "Too much vibration",
                 "meaning": "The drone was shaking more than it should. This usually comes from a damaged or unbalanced "
                            "propeller, a loose motor, or a loose flight-controller mount, and it can confuse the sensors.",
                 "check": "Check every propeller for chips, cracks or bends, and make sure motors and the flight controller are tight.",
                 "check_critical": "Do not fly until the propellers, motors and flight-controller mount have been checked and fixed."},
    "SENSOR_FAIL": {"name": "A sensor stopped working properly",
                    "meaning": "One of the drone's sensors stopped sending data, froze, or sent values that make no sense.",
                    "check": "Check the sensor's cable and connector, and recalibrate it before the next flight.",
                    "check_critical": "Do not fly until the failed sensor has been checked or replaced."},
    "GPS_DEGRADED": {"name": "Weak or lost GPS",
                     "meaning": "The GPS position was poor or lost for part of the flight, so the drone may have drifted "
                                "or not held its position.",
                     "check": "Fly in an open area away from buildings and metal, and check the GPS antenna and cable.",
                     "check_critical": "Do not fly in GPS modes until the GPS problem is understood."},
    "EST_DIVERGE": {"name": "The drone lost track of where it was",
                    "meaning": "The drone's position and heading estimate became unreliable or had to reset during the flight.",
                    "check": "Recalibrate the compass away from metal, and check the GPS and the compass placement.",
                    "check_critical": "Do not fly until the compass and GPS have been recalibrated and checked."},
    "CTRL_OSC": {"name": "Wobbling or poor control",
                 "meaning": "The drone did not follow its controls smoothly: it wobbled or lagged behind what it was asked to do.",
                 "check": "Check propellers and motors first; if they are fine, the control tuning may need adjusting.",
                 "check_critical": "Do not fly until the cause of the wobble has been found."},
    "CTRL_SAT": {"name": "Motors at full power",
                 "meaning": "One or more motors ran at full power for a while. The drone had no power left in reserve.",
                 "check": "Check the weight of the drone and payload, the battery charge, and that no motor or propeller is weak.",
                 "check_critical": "Do not fly until the drone is lighter or the weak motor has been found."},
    "MOTOR_ESC": {"name": "A motor or propeller is not working like the others",
                  "meaning": "One motor needed a different amount of power from the others to keep the drone level. This "
                             "points to a damaged propeller, a worn motor, a speed controller problem, or uneven weight.",
                  "check": "Inspect that motor and its propeller, spin the motor by hand for roughness, and check the weight balance.",
                  "check_critical": "Do not fly until that motor, its propeller and its speed controller have been checked."},
    "POWER": {"name": "Battery or power problem",
              "meaning": "The battery voltage dropped too low or sagged a lot under load, or the electronics supply dipped.",
              "check": "Charge and check the battery, look for swelling or a weak cell, and check the power connectors.",
              "check_critical": "Do not fly with this battery until it has been checked; land earlier next time."},
    "TIMING": {"name": "Flight computer running late",
               "meaning": "The flight computer did not always finish its work on time.",
               "check": "Report this to the engineering team before the next flight.",
               "check_critical": "Do not fly until engineering has checked the flight computer."},
    "COMMS": {"name": "Radio link lost",
              "meaning": "The connection to the remote control or ground station dropped during the flight.",
              "check": "Check the antennas and radio range, and keep the drone closer next time.",
              "check_critical": "Do not fly until the radio link has been range-tested."},
    "SECURITY": {"name": "Suspicious commands",
                 "meaning": "The drone rejected commands that did not come from an approved source.",
                 "check": "Report this to the engineering team.",
                 "check_critical": "Do not fly until engineering has checked the command link."},
    "LOGGING": {"name": "Gaps in the flight recording",
                "meaning": "Some data was not recorded, so parts of the flight cannot be checked.",
                "check": "Check the SD card and replace it if it is old or slow.",
                "check_critical": "Replace the SD card before the next flight."},
    "CRASH": {"name": "Crash or loss of control",
              "meaning": "The drone tipped over, hit something, or the flight controller detected a crash.",
              "check": "Inspect the whole drone for damage: frame, arms, propellers, motors, battery and landing gear.",
              "check_critical": "Do not fly until the whole drone has been inspected for crash damage."},
}

NOT_CHECKED_PLAIN = {
    "VIB_HIGH": "vibration", "SENSOR_FAIL": "sensor health", "GPS_DEGRADED": "GPS", "EST_DIVERGE": "position tracking",
    "CTRL_OSC": "how smoothly it flew", "CTRL_SAT": "motor power reserve", "MOTOR_ESC": "motor balance", "POWER": "battery",
    "TIMING": "flight-computer timing", "COMMS": "radio link", "SECURITY": "command security", "LOGGING": "recording gaps",
    "CRASH": "crash detection",
}


def _when(f: Dict[str, Any], takeoff: Optional[float], landing: Optional[float]) -> str:
    if f.get("window") == "whole_flight":
        return "throughout the flight"
    s, e = f.get("t_start"), f.get("t_end")
    if s is None:
        return ""
    where = ""
    if takeoff is not None and s - takeoff <= 20:
        where = "just after take-off"
    elif landing is not None and landing - (e or s) <= 30:
        where = "near the end of the flight"
    else:
        where = "during the flight"
    return f"{where} ({clock(s)} into the recording)"


def _headline(f: Dict[str, Any]) -> str:
    """A short plain sentence for one finding, using its most telling number where it helps."""
    code, det, ev = f["code"], f.get("detector", ""), f.get("evidence") or [{}]
    v = ev[0].get("value")
    title = f.get("title", "")
    if det == "rule:tilt" and v is not None:
        return f"The drone tilted to {v:.0f}° (it should stay well under 60°)."
    if det == "event_log":
        return title.replace("Flight event:", "The flight controller reported:").rstrip(".") + "."
    if code == "VIB_HIGH" and "clip" in det:
        return "The vibration was so strong that the motion sensor hit its limit."
    if code == "VIB_HIGH" and v is not None:
        return f"Vibration reached {v:.0f} m/s² (the limit is 30)."
    if det == "physics:hover_balance":
        return title.replace("works harder in hover", "had to work harder than the others to hover").replace(
            "works less in hover", "needed less power than the others to hover") + "."
    if code == "POWER" and "per cell" in title:
        return title.replace("Battery fell to", "The battery dropped to") + " (a healthy cell stays above 3.2 V)."
    if code == "CTRL_SAT":
        return title.replace("at maximum output for", "was at full power for") + "."
    if code == "CTRL_OSC":
        return "The drone did not hold its angle smoothly: " + ("roll" if "roll" in title else "pitch") + " kept overshooting."
    if code == "SENSOR_FAIL" and "frozen" in title:
        return title.split(" frozen")[0].split(" ")[0].replace("_", " ").capitalize() + " sensor values stopped changing for a while."
    if code == "SENSOR_FAIL" and "stopped updating" in title:
        return "A sensor stopped sending data for a moment."
    return PLAIN.get(code, {}).get("name", code) + "."


def _count(n: int, noun: str) -> str:
    words = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five"}
    return f"{words.get(n, n)} {noun}{'' if n == 1 else 's'}"


def _action(f: Dict[str, Any], p: Dict[str, str]) -> str:
    action = p["check_critical"] if f["severity"] == "critical" else p["check"]
    if f["code"] == "MOTOR_ESC" and f.get("detector", "").startswith("physics"):
        motor = f.get("title", "").split(" works")[0].split(" makes")[0]
        action = action.replace("that motor", motor)
    return action


def pilot_view(a: Dict[str, Any], agent_report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The plain-language briefing shown at the top of the page."""
    if a.get("refused"):
        return {"status": "unreadable", "status_text": "This log could not be read",
                "status_detail": "It uses a format or settings this app does not know, so it was not analysed. "
                                 "Send it to the engineering team.", "problems": [], "checklist": [], "not_checked": []}
    cls = a.get("classification") or {}
    tl = a.get("timeline") or {}
    takeoff, landing = tl.get("takeoff_t"), tl.get("landing_t")
    findings = a.get("findings") or []
    reviews = {r["finding_id"]: r for r in ((agent_report or {}).get("finding_reviews") or [])}

    problems = []
    for f in findings:
        p = PLAIN.get(f["code"], {"name": f["code"], "meaning": "", "check": "", "check_critical": ""})
        rv = reviews.get(f["id"]) or {}
        problems.append({
            "finding_id": f["id"], "severity": f["severity"], "code": f["code"], "name": p["name"],
            "headline": _headline(f), "when": _when(f, takeoff, landing), "meaning": p["meaning"],
            "action": _action(f, p),
            "likely_false_alarm": rv.get("assessment") == "likely_false_alarm",
            "confidence": f.get("confidence"),
        })
    # one card per kind of problem: the most severe finding leads, the rest are counted
    grouped: Dict[str, Dict[str, Any]] = {}
    for p in problems:
        g = grouped.get(p["code"])
        if g is None:
            grouped[p["code"]] = {**p, "also": []}
        else:
            g["also"].append(p["headline"])
    cards = sorted(grouped.values(), key=lambda p: (p["severity"] != "critical", p["likely_false_alarm"]))

    critical_checks: List[str] = []
    other_checks: List[str] = []
    for c in cards:
        if not c["likely_false_alarm"]:
            target = critical_checks if c["severity"] == "critical" else other_checks
            if c["action"] not in critical_checks + other_checks:
                target.append(c["action"])

    real = [c for c in cards if not c["likely_false_alarm"]]
    if cls.get("kind") and cls["kind"] != "flight":
        status, text = "not_a_flight", "Not a full flight"
        detail = ("The drone was not in the air long enough to check the flight. " if cls["kind"] == "ground_only"
                  else "The motors ran but the drone did not move (a bench or tethered test). ") + \
                 "Only the recording itself was checked."
    elif any(c["severity"] == "critical" for c in real):
        status, text = "critical", "Do not fly again until this is checked"
        n = sum(c["severity"] == "critical" for c in real)
        detail = f"{_count(n, 'serious problem')} found. Inspect the drone before the next flight."
    elif real:
        status, text = "warning", "Check the drone before the next flight"
        detail = f"{_count(len(real), 'problem')} worth checking. None of them is serious on its own."
    else:
        status, text = "clear", "No problems found"
        detail = "Nothing went wrong in the parts of the flight that could be checked. This is not a guarantee that the drone is safe."

    not_checked = [NOT_CHECKED_PLAIN[c] for c, v in (a.get("coverage") or {}).items()
                   if v.get("status") in ("not_evaluable", "partial") and c in NOT_CHECKED_PLAIN and c not in ("TIMING", "SECURITY")]
    integrity = (a.get("integrity") or {}).get("result")
    warning = "This recording has been changed after the flight, so these results cannot be fully trusted." \
        if integrity == "modified" else None
    return {"status": status, "status_text": text, "status_detail": detail, "problems": cards,
            "critical_checks": critical_checks, "checklist": critical_checks + other_checks,
            "not_checked": not_checked, "integrity_warning": warning,
            "flight": {"duration_s": tl.get("duration_s"), "airborne_s": tl.get("airborne_s")}}
