"""Pilot-facing translation of stored analysis evidence.

This module only *re-words* evidence that already exists in the deterministic event
records and in the deployed detector's report. It does not score, re-threshold,
merge, split or otherwise change any event, and it never asks a model anything.

The goal is a flight briefing a pilot can read in about a minute:

* a short flight summary,
* the few periods that deserve attention (using the existing review priority),
* the important changes in ordinary language,
* practical review steps,
* one concise limitation statement.

Engineering evidence stays available in the caller's technical appendix. Raw
channel identifiers belong there, not in the briefing.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from src.flight_relationships import (
    INTERPRETATION_LABEL,
    command_sentence,
    consistency_sentence,
    physical_summary,
)

LIMITATION = (
    "Some recorded values cannot yet be tied to a specific aircraft part or unit. Their changes are "
    "reported, but their physical meaning is not established."
)

REVIEW_LABEL = {
    "review_first": "Check this first",
    "review": "Worth checking",
    "minor": "For reference",
}

REVIEW_ORDER = {"review_first": 0, "review": 1, "minor": 2}

GROUP_NAMES = {
    "motor_output": ("motor output", "motor outputs"),
    "gyroscope": ("rotation reading", "rotation readings"),
    "accelerometer": ("movement reading", "movement readings"),
    "attitude": ("attitude reading", "attitude readings"),
    "distance_height": ("height reading", "height readings"),
    "controller_internal": ("controller reading", "controller readings"),
    "pilot_command": ("stick or switch input", "stick or switch inputs"),
    "status_flag": ("status flag", "status flags"),
    "unclassified": ("unidentified telemetry reading", "unidentified telemetry readings"),
}

# Friendly names are only used where the deployed detector's signal meaning is
# documented. Any other channel stays an unidentified telemetry reading.
DEEP_SIGNAL_LABELS = {
    "gyro_x": "rotation about X",
    "gyro_y": "rotation about Y",
    "gyro_z": "rotation about Z",
    "acc_x": "movement along X",
    "acc_y": "movement along Y",
    "acc_z": "movement along Z",
    "roll": "roll angle",
    "pitch": "pitch angle",
    "yaw_sin": "heading (sine component)",
    "yaw_cos": "heading (cosine component)",
    "roll_res": "roll tracking difference",
    "pitch_res": "pitch tracking difference",
}

_WORD = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}


def _word(n: int) -> str:
    return _WORD.get(int(n), str(int(n)))


def _cap(text: str) -> str:
    """Capitalise the first character without lower-casing the rest (keeps 'X', 'Z')."""
    return text[:1].upper() + text[1:] if text else text


def _join_items(items: List[str]) -> str:
    """Natural-language list: 'a', 'a and b', 'a, b and c'."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + " and " + items[-1]


def _rows(ev: Dict[str, Any]) -> str:
    if ev.get("start_row") is None or ev.get("end_row") is None:
        return "the selected interval"
    if ev["start_row"] == ev["end_row"]:
        return f"row {ev['start_row']}"
    return f"rows {ev['start_row']}–{ev['end_row']}"


def _int(n: Any) -> str:
    try:
        return f"{int(round(float(n))):,}"
    except (TypeError, ValueError):
        return str(n)


def _clock(seconds: float) -> str:
    s = max(0, int(round(seconds)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def _elapsed(seconds: float) -> str:
    """Readable interval; seconds for short flights, clock form for longer ones."""
    value = max(0.0, float(seconds))
    if value < 120:
        return f"{value:.1f} s"
    return _clock(value)


_UNIT_SCALE = {"ms": 1e-3, "s": 1.0, "us": 1e-6, "ns": 1e-9}


def _rel_seconds(raw: Any, timing: Dict[str, Any]) -> Optional[float]:
    """Seconds after the start of the recording, only when the time unit is known (verified or assumed)."""
    if raw is None:
        return None
    unit = str(((timing.get("time_basis") or {}).get("unit_hypothesis")) or "").lower()
    scale = _UNIT_SCALE.get(unit)
    first = timing.get("first_raw")
    if scale is None or first is None:
        return None
    try:
        return (float(raw) - float(first)) * scale
    except (TypeError, ValueError):
        return None


def _time_unit_verified(timing: Dict[str, Any]) -> bool:
    return (timing.get("time_basis") or {}).get("status") == "verified"


def _ratio(before: Any, during: Any) -> Optional[float]:
    """How many times the earlier level a measurement reached; unit-free, so it works for unverified channels."""
    try:
        b, d = float(before), float(during)
    except (TypeError, ValueError):
        return None
    if b == 0 or d <= 0 or b <= 0:
        return None
    r = d / b
    return r if 0.05 <= r <= 20 else None


def _magnitude_text(ratios: List[float], pronoun: str = "their") -> str:
    """'about 1.4× their earlier level' / 'about 30% below their earlier level'. Unit-free and honest."""
    if not ratios:
        return ""
    lo, hi = min(ratios), max(ratios)
    if len(ratios) > 1 and (hi - lo) >= max(0.15 * lo, 0.2):
        return f"about {lo:.1f}–{hi:.1f}× {pronoun} earlier level"
    mid = (lo + hi) / 2
    if mid >= 3:
        return f"about {mid:.0f}× {pronoun} earlier level"
    if mid >= 1.08:
        return f"about {mid:.1f}× {pronoun} earlier level"
    if mid <= 0.92:
        return f"about {(1 - mid) * 100:.0f}% below {pronoun} earlier level"
    return f"about the same as {pronoun} earlier level"


def _measurement_magnitude(m: Dict[str, Any]) -> Optional[float]:
    before, during = m.get("before") or {}, m.get("during") or {}
    if not before.get("available"):
        return None
    b = before.get("median")
    span = max(abs(during.get("min") or 0.0), abs(during.get("max") or 0.0), 1e-12)
    # A percentage or ratio against a near-zero baseline says nothing useful.
    if b is None or abs(float(b)) < 0.05 * span:
        return None
    return _ratio(b, during.get("median"))


def review_label(priority: Optional[str]) -> str:
    return REVIEW_LABEL.get(str(priority or "review"), "Worth checking")


def group_phrase(group: Optional[str], count: int, total: Optional[int] = None) -> str:
    singular, plural = GROUP_NAMES.get(str(group or "unclassified"), GROUP_NAMES["unclassified"])
    if count <= 1:
        return f"The {singular}"
    if total is not None and count == total:
        return f"All {_word(count)} {plural}" if count > 2 else f"Both {plural}"
    if total is not None and count < total:
        return f"{_word(count).capitalize()} of the {_word(total)} {plural}"
    return f"{_word(count).capitalize()} {plural}"


def friendly_signal_label(signal: str, *, group: Optional[str] = None, deep: bool = False) -> str:
    """Plain label for a chart or sentence; the actual column name stays in technical data."""
    if deep and signal in DEEP_SIGNAL_LABELS:
        return DEEP_SIGNAL_LABELS[signal]
    _singular, plural = GROUP_NAMES.get(str(group or "unclassified"), GROUP_NAMES["unclassified"])
    return f"Unidentified telemetry reading" if group in (None, "unclassified") else plural.capitalize()


def _sustained(ev: Dict[str, Any]) -> bool:
    return any(str((m.get("persistence") or "")).startswith("sustained") for m in ev.get("measurements", []))


def _simultaneous(ev: Dict[str, Any]) -> bool:
    return any(a.get("type") in ("simultaneous", "group") for a in ev.get("associations", []))


def _changed_groups(ev: Dict[str, Any]) -> List[Dict[str, Any]]:
    order = [
        "motor_output",
        "distance_height",
        "attitude",
        "gyroscope",
        "accelerometer",
        "controller_internal",
        "pilot_command",
        "status_flag",
    ]
    groups = [a for a in ev.get("associations", []) if a.get("type") == "group" and a.get("channels")]
    groups.sort(
        key=lambda a: (
            0 if len(a["channels"]) == a.get("n_in_file", len(a["channels"])) else 1,
            order.index(a["group"]) if a.get("group") in order else 99,
            -(len(a.get("channels") or [])),
        )
    )
    return groups


def _group_ratios(a: Dict[str, Any], ev: Dict[str, Any]) -> List[float]:
    by_ch = {m.get("channel"): m for m in ev.get("measurements") or []}
    out: List[float] = []
    for ch in a.get("channels") or []:
        m = by_ch.get(ch)
        r = _measurement_magnitude(m) if m else None
        if r is not None:
            out.append(r)
    return out


def _group_sentence(a: Dict[str, Any], ev: Optional[Dict[str, Any]] = None) -> str:
    count = len(a.get("channels") or [])
    total = int(a.get("n_in_file") or count)
    who = group_phrase(a.get("group"), count, total)
    direction = a.get("direction")
    mag = _magnitude_text(_group_ratios(a, ev) if ev else [])
    tail = f" ({mag})" if mag else ""
    # A tiny median shift against a much larger swing is a transient, not a new level.
    if ev is not None:
        by = {m.get("channel"): m for m in ev.get("measurements") or []}
        shifts = [abs(by[c].get("shift") or 0.0) for c in a.get("channels") or [] if c in by]
        swings = [abs(by[c].get("swing") or 0.0) for c in a.get("channels") or [] if c in by]
        if shifts and swings and max(shifts) < 0.15 * max(swings) and max(swings) > 0:
            return f"{who} changed briefly at about the same time, without settling at a new level."
    if direction == "higher":
        verb = "was higher" if count == 1 else "were higher"
        return f"{who} {verb} than earlier in the recording{tail}."
    if direction == "lower":
        verb = "was lower" if count == 1 else "were lower"
        return f"{who} {verb} than earlier in the recording{tail}."
    verb = "changed" if count == 1 else "changed"
    return f"{who} {verb} at about the same time, in different directions."


def _measurement_group(m: Dict[str, Any]) -> str:
    return str(m.get("group") or ("status_flag" if m.get("role") == "status" else "unclassified"))


def _singular_group(group: str) -> str:
    return GROUP_NAMES.get(group, GROUP_NAMES["unclassified"])[0]


def _event_group_sentences(ev: Dict[str, Any], limit: int = 3) -> List[str]:
    groups = _changed_groups(ev)
    out = [_group_sentence(a, ev) for a in groups[:limit]]
    if out:
        return out

    # No co-movement group was recorded: describe the individual measurements
    # without inventing a relationship between them.
    measurements = [m for m in (ev.get("measurements") or []) if m.get("channel")]
    if not measurements:
        return out
    seen: List[str] = []
    for m in measurements:
        g = _measurement_group(m)
        if g not in seen:
            seen.append(g)
    if len(measurements) == 1:
        m = measurements[0]
        label = _singular_group(_measurement_group(m))
        mag = _magnitude_text([r] if (r := _measurement_magnitude(m)) else [])
        tail = f" ({mag})" if mag else ""
        if m.get("direction") == "higher":
            out.append(f"The {label} was higher than earlier in the recording{tail}.")
        elif m.get("direction") == "lower":
            out.append(f"The {label} was lower than earlier in the recording{tail}.")
        else:
            out.append(f"The {label} moved away from its earlier pattern.")
        return out
    if len(seen) == 1:
        n = len(measurements)
        label = GROUP_NAMES.get(seen[0], GROUP_NAMES["unclassified"])[1]
        directions = {str(m.get("direction") or "") for m in measurements}
        mag = _magnitude_text([r for r in (_measurement_magnitude(m) for m in measurements) if r])
        tail = f" ({mag})" if mag else ""
        if directions == {"higher"}:
            out.append(f"{_word(n).capitalize()} {label} were higher than earlier in the recording{tail}.")
        elif directions == {"lower"}:
            out.append(f"{_word(n).capitalize()} {label} were lower than earlier in the recording{tail}.")
        else:
            out.append(f"{_word(n).capitalize()} {label} moved away from their earlier pattern.")
        return out
    labels = [_singular_group(g) for g in seen[:3]]
    if len(labels) == 2:
        subject = f"a {labels[0]} and a {labels[1]}"
    else:
        subject = ", ".join(f"a {x}" for x in labels[:-1]) + f" and a {labels[-1]}"
    out.append(f"{_cap(subject)} moved away from their earlier pattern in this interval.")
    return out


def _has_kind(ev: Dict[str, Any], kind: str) -> bool:
    return kind in (ev.get("kinds") or [])


def _what_happened(ev: Dict[str, Any]) -> str:
    cat = ev.get("category")
    if cat == "data_quality":
        if _has_kind(ev, "sampling_gap"):
            return "The recording has a gap in this interval, so the system could not see what happened between the two sides of it."
        if _has_kind(ev, "missing_values"):
            return "Some recorded values are missing in this interval, so the system could not assess all of it."
        if _has_kind(ev, "flatline"):
            return "One or more recorded values stopped changing and held the same value here, although they normally vary."
        return "The recording is irregular in this interval, so it could not be fully assessed."

    if cat == "state_transition":
        parts: List[str] = []
        for m in ev.get("measurements", [])[:3]:
            g = m.get("group") or ("status_flag" if m.get("role") == "status" else "unclassified")
            if g == "pilot_command":
                parts.append("a stick or switch input")
            elif g == "status_flag":
                parts.append("a recorded status value")
            else:
                parts.append(GROUP_NAMES.get(g, GROUP_NAMES["unclassified"])[0])
        uniq = list(dict.fromkeys(parts)) or ["a recorded status value"]
        if len(uniq) == 1:
            subject = uniq[0]
        else:
            subject = ", ".join(uniq[:-1]) + " and " + uniq[-1]
        return f"{_cap(subject)} changed in this interval. These are recorded inputs or states, not measured aircraft motion."

    sentences = _event_group_sentences(ev)
    if sentences:
        text = " ".join(sentences[:2])
        if _sustained(ev) and "briefly" not in text:
            text += " The different levels were still present later in the recording."
        if _simultaneous(ev):
            text += " These changes began close together."
        return text

    n = int(ev.get("n_channels_involved") or len(ev.get("channels") or []) or 1)
    noun = "reading" if n == 1 else "readings"
    return f"{_word(n).capitalize()} telemetry {noun} moved away from the earlier pattern in this interval."


def _why_review(ev: Dict[str, Any], technical_why: Optional[str] = None) -> str:
    interp = ev.get("interpretation")
    cat = ev.get("category")
    if cat == "data_quality":
        base = ("The recording itself is incomplete or irregular here, which limits what can be concluded "
                "about this period.")
    elif interp == "control_input_change":
        base = ("A recorded input or status changed here. It is listed so it is not mistaken for aircraft "
                "behaviour; it is not a deviation of the aircraft itself.")
    elif interp == "telemetry_inconsistency":
        base = ("Two records of the same motion disagree here. One of them may be disturbed, but the file "
                "cannot say which.")
    elif interp == "expected_response":
        base = "This change follows the recorded input and is listed for completeness."
    elif interp == "unexplained_deviation":
        base = ("No matching pilot input or companion measurement explains this change, so it is the part "
                "worth reviewing.")
    elif interp == "insufficient_evidence":
        base = "The change is small or the comparison window is too short to interpret it."
    else:
        base = "This interval is listed for completeness."
    if _sustained(ev):
        base += " The new level did not immediately return to the earlier pattern."
    if _simultaneous(ev) and cat != "data_quality":
        base += " The changes happened close together, but the file cannot show that one caused another."
    return base


def _time_span(start: float, end: float) -> str:
    """One clock value for a very short period, a range otherwise; avoids '00:04–00:04'."""
    if end - start < 1.0:
        return _clock(start)
    return f"{_clock(start)}–{_clock(end)}"


def _when_direct(ev: Dict[str, Any], timing: Dict[str, Any]) -> Dict[str, Any]:
    start_raw, end_raw = ev.get("start_raw"), ev.get("end_raw")
    start_rel = _rel_seconds(start_raw, timing)
    end_rel = _rel_seconds(end_raw, timing)
    out: Dict[str, Any] = {
        "start_row": ev.get("start_row"),
        "end_row": ev.get("end_row"),
        "rows": _rows(ev),
        "start_raw": start_raw,
        "end_raw": end_raw,
        "start_rel_s": start_rel,
        "end_rel_s": end_rel,
    }
    if start_rel is not None and end_rel is not None:
        verified = _time_unit_verified(timing)
        span = _time_span(start_rel, end_rel)
        out["short"] = span
        out["display"] = f"{span} into the flight"
        out["approximate"] = "" if verified else "Approximate: the file's time unit is not verified."
    else:
        out["short"] = _rows(ev)
        out["display"] = f"{_rows(ev)} (the file has no usable time column)"
        out["approximate"] = ""
    return out


def _when_deep(detail: Dict[str, Any]) -> Dict[str, Any]:
    start, end = detail.get("start_label"), detail.get("end_label")
    duration = detail.get("duration_s")
    display = f"{start}–{end} into the flight" + (f" ({duration:.1f} s)" if duration is not None else "")
    return {
        "display": display,
        "approximate": "",
        "rows": None,
        "start_row": None,
        "end_row": None,
        "start_rel_s": detail.get("start_rel_s"),
        "end_rel_s": detail.get("end_rel_s"),
    }


def _what_to_check(ev: Dict[str, Any], when: Dict[str, Any]) -> List[str]:
    where = when.get("short") or when.get("display", "this period")
    cat = ev.get("category")
    interp = ev.get("interpretation")
    if cat == "data_quality":
        return [
            f"Replay the flight around {where} and check what the aircraft was doing while the data was missing.",
            "If that part of the flight matters, compare it with the original flight log or a fresh download.",
        ]
    if interp == "telemetry_inconsistency":
        return [
            f"Compare the attitude, rotation-rate and acceleration records around {where} in the original log.",
            "Check whether the aircraft was disturbed (for example a hard contact) at that moment.",
        ]
    if (ev.get("family") or "") == "acceleration":
        return [
            f"Replay the flight around {where} and check for a manoeuvre, contact or rough air at that moment.",
            "Compare the acceleration transient with the attitude and rotation-rate records for the same period.",
        ]
    if (ev.get("family") or "") in ("altitude", "distance"):
        return [
            f"Compare the estimated height, the height setpoint and the time-of-flight reading around {where}.",
            "Check whether the height change was intended and whether the two height sources stayed consistent.",
        ]
    return [
        f"Replay the flight around {where} and note what the aircraft was doing.",
        "Compare it with the pilot's inputs and whether it was expected for that part of the flight.",
    ]


def _event_sentence(ev: Dict[str, Any], timing: Dict[str, Any]) -> str:
    when = _when_direct(ev, timing)["display"]
    headline = str(ev.get("headline") or ev.get("interpretation_label") or "a change")
    return f"The main period is at {when}: {headline[:1].lower() + headline[1:]}."


def _system_check_summary(rid: str, rel: Dict[str, Any]) -> str:
    status = rel.get("status")
    r = rel.get("r")
    if rid == "roll_tracking":
        if status == "consistent":
            return "Roll input and roll angle follow each other at manoeuvre scale."
        if status == "partially_consistent":
            return "Roll input and roll angle partly follow each other."
        if status == "insufficient_data":
            return "The roll input barely varied, so tracking could not be established."
        return "Roll input and roll angle do not follow each other in this recording."
    if rid == "pitch_tracking":
        if status == "consistent":
            return "Pitch input and pitch angle follow each other at manoeuvre scale."
        if status == "insufficient_data":
            return "The pitch input barely varied, so tracking could not be established."
        return "Pitch input and pitch angle do not follow each other closely in this recording."
    if rid == "throttle_motors":
        return ("Motor output follows the throttle input at manoeuvre scale." if status in ("consistent", "partially_consistent")
                else "Motor output does not follow the throttle input closely in this recording.")
    if rid == "altitude_setpoint":
        return ("The altitude estimate follows the altitude setpoint." if status in ("consistent", "partially_consistent")
                else "The altitude estimate does not follow the setpoint closely.")
    if rid == "altitude_independent_measurement":
        base = ("The altitude estimate and the time-of-flight height stay at the same level and move together."
                if status in ("consistent", "partially_consistent") else
                "The altitude estimate and the time-of-flight height do not agree.")
        if r is not None and abs(r) > 0.999:
            base += " Their correlation is nearly perfect, so they may not be independent measurements."
        return base
    if rid.startswith("attitude_rate"):
        return ("The attitude change matches the measured rotation rate."
                if status == "consistent" else
                "The attitude change and the measured rotation rate only partly match."
                if status == "partially_consistent" else
                "The attitude change and the measured rotation rate do not match.")
    if rid == "gravity":
        if rel.get("rest_median") is not None:
            return (f"At rest the acceleration magnitude is {rel['rest_median']:.2f} m/s^2, consistent with gravity."
                    if status == "consistent" else
                    f"At rest the acceleration magnitude is {rel['rest_median']:.2f} m/s^2, which does not match gravity.")
        return "There were too few at-rest samples to check acceleration against gravity."
    if rid == "motor_balance":
        p95 = rel.get("p95_spread_fraction")
        return ("The four motor outputs stay within a small spread of each other."
                if status == "consistent" else
                f"The motor outputs differ from each other at times (95th percentile spread {p95:.0%} of their mean)."
                if isinstance(p95, (int, float)) else "Motor balance could not be assessed.")
    return rel.get("detail", "")


def _duration_text(timing: Dict[str, Any], rows: int) -> str:
    dur = timing.get("duration_s") or {}
    if dur.get("value") is not None:
        value = float(dur["value"])
        basis = str(dur.get("basis") or "")
        if basis == "verified":
            return f"The flight recording covers about {value:.1f} s"
        unit = str(timing.get("time_basis", {}).get("unit_hypothesis") or "")
        label = "milliseconds" if unit == "ms" else (unit or "an unknown unit")
        return f"The flight recording covers about {value:.1f} s, assuming the time column is in {label} (not verified)"
    return f"The file has {_int(rows)} rows and no usable time column"


def _observation_sentences(ev: Dict[str, Any], timing: Dict[str, Any]) -> List[str]:
    cat = ev.get("category")
    when = _when_direct(ev, timing)
    if cat == "data_quality":
        return [f"{when['display']}: part of the recording could not be assessed."]
    out = _event_group_sentences(ev, limit=3)
    if _sustained(ev) and out:
        out[-1] = out[-1].rstrip(".") + " and stayed at the new level."
    if not out:
        out = [f"{when['display']}: the recorded telemetry changed away from its earlier pattern."]
    return out


def _next_steps(events: List[Dict[str, Any]], timing: Dict[str, Any]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen = set()

    def add(text: str, event_id: str = "") -> None:
        key = text.strip().lower()
        if key and key not in seen and len(out) < 5:
            seen.add(key)
            out.append({"event_id": event_id, "text": text})

    for ev in events[:3]:
        when = _when_direct(ev, timing)
        where = when.get("short") or when.get("display", "this period")
        if ev.get("category") == "data_quality":
            add(f"Replay the flight around {where} and check what the aircraft was doing while the data was missing.", ev.get("id", ""))
        elif (ev.get("family") or "") == "acceleration":
            add(f"Replay the flight around {where} and check for a manoeuvre, contact or rough air.", ev.get("id", ""))
        elif (ev.get("family") or "") in ("altitude", "distance"):
            add(f"Compare the estimated height, setpoint and time-of-flight reading around {where}.", ev.get("id", ""))
        else:
            add(f"Replay the flight around {where} and note what the aircraft was doing.", ev.get("id", ""))
    add("Compare these changes with what the pilot was doing at the time.")
    add("Check whether the behaviour was expected for that part of the flight.")
    add("Open the original flight log if you need the exact values.")
    return out


def build_direct_pilot_report(data: Dict[str, Any]) -> Dict[str, Any]:
    """Pilot briefing for *AI Analysis*: deterministic, context-aware, no model involved."""
    timing = data.get("timing") or {}
    flight = data.get("flight") or {}
    events = list(data.get("events") or [])
    ranked = sorted(events, key=lambda e: (REVIEW_ORDER.get(e.get("priority"), 1), e.get("start_index", 0)))
    # Files without a telemetry schema keep the earlier statistical briefing: significance is unknown, not small.
    physical = [e for e in ranked if e.get("significance") in ("physical", None)]
    review = [e for e in physical if not e.get("expected") and e.get("category") != "state_transition"]
    aircraft_families = {"attitude", "acceleration", "angular_rate", "altitude", "motor"}
    consistent = [e for e in physical if e.get("expected") and e.get("category") != "state_transition"
                  and (e.get("family") or "") in aircraft_families]
    quality = [e for e in ranked if e.get("category") == "data_quality"]
    attention_events = (quality + review)[:3]
    if not attention_events and not flight and ranked:
        # No schema means no physical significance test: keep the earlier statistical briefing.
        attention_events = ranked[:2]
    attention_ids = {e.get("id") for e in attention_events}
    consistent_ids = {e.get("id") for e in consistent}
    other = [e for e in ranked if e.get("id") not in attention_ids and e.get("id") not in consistent_ids
             and e.get("category") != "state_transition"
             and (e.get("family") or "") in aircraft_families
             and e.get("significance") in ("physical", None)][:8]

    duration_text = _duration_text(timing, int(data.get("data_quality", {}).get("rows") or 0))
    clusters = _time_clusters(attention_events)
    if attention_events:
        n = len(quality) + len(review)
        if len(clusters) == 1 and len(attention_events) > 1:
            human = {"attitude": "attitude", "acceleration": "acceleration", "angular_rate": "rotation-rate",
                     "altitude": "altitude", "motor": "motor-output", "control": "controller"}
            groups = _join_items([human.get(str(e.get("family") or ""), str(e.get("family") or "")) for e in attention_events[:3]])
            summary = (
                f"{duration_text}. One period stands out at {_when_direct(attention_events[0], timing)['display']}: "
                f"the {groups} records all changed there and the recorded input does not explain the change. "
            )
            if consistent:
                summary += (f"{len(consistent)} other change{'s' if len(consistent) != 1 else ''} "
                            f"{'match' if len(consistent) != 1 else 'matches'} the recorded input. ")
            summary += "The recording shows what changed, but not why."
        else:
            summary = (
                f"{duration_text}. "
                f"{n} period{'s' if n != 1 else ''} {'are' if n != 1 else 'is'} worth reviewing. "
                f"{_event_sentence(attention_events[0], timing)} "
            )
            if consistent:
                summary += (f"{len(consistent)} further change{'s' if len(consistent) != 1 else ''} "
                            f"{'match' if len(consistent) != 1 else 'matches'} the recorded input. ")
            summary += "The recording shows what changed, but not why."
        bottom_line = (f"Start with the {len(quality) + len(review)} period"
                       f"{'s' if (len(quality) + len(review)) != 1 else ''} below. "
                       "These are changes in the recording, not a diagnosis.")
    else:
        summary = (
            f"{duration_text}. No unexplained deviation was found in the part of the recording that could be "
            "assessed. "
            + (f"{len(consistent)} change{'s' if len(consistent) != 1 else ''} "
               f"{'were' if len(consistent) != 1 else 'was'} consistent with the recorded input. "
               if consistent else "")
            + "That does not confirm that the flight was normal."
        )
        bottom_line = "No unexplained deviation stood out. That is not a clean bill of health."

    observations: List[str] = []
    for ev in attention_events[:3]:
        for sentence in _observation_sentences(ev, timing):
            if sentence not in observations:
                observations.append(sentence)
        for p in (ev.get("physical_values") or []):
            if p.get("category") not in aircraft_families:
                continue
            if len([x for x in observations if not x.startswith("No ")]) >= 4:
                break
            text = physical_summary(p)
            if text and text not in observations:
                observations.append(text)
    if not observations:
        observations.append("No change in the assessed part of the recording needed explanation.")
    observations = observations[:6]

    system_checks = []
    for rid, rel in (flight.get("relationships") or {}).items():
        system_checks.append({
            "id": rid, "label": rel.get("label", rid), "status": rel.get("status"),
            "summary": _system_check_summary(rid, rel), "r": rel.get("r"),
        })

    overview = {
        "duration_s": (timing.get("duration_s") or {}).get("value"),
        "sample_rate_hz": (timing.get("sampling_hz") or {}).get("value"),
        "time_verified": _time_unit_verified(timing),
        "rows": int(data.get("data_quality", {}).get("rows") or 0),
        "columns": int(data.get("data_quality", {}).get("columns") or 0),
        "signals_available": len((flight.get("signals") or {}).get("available") or []),
        "events_total": len(events),
        "events_to_review": len(quality) + len(review),
        "events_consistent": len(consistent),
        "flight_signals": sum(len(v) for k, v in (flight.get("signals") or {}).get("by_category", {}).items()
                              if k in aircraft_families),
        "status_changes": len(flight.get("status_transitions") or []),
        "sequence": list(flight.get("sequence") or []),
        "data_quality": {
            "missing_signals": len((flight.get("data_quality") or {}).get("missing_values") or {}),
            "gaps": int((timing.get("n_gaps") or 0)),
            "out_of_range": len((flight.get("data_quality") or {}).get("out_of_range") or []),
            "short_recording": bool((flight.get("timing") or {}).get("short_recording")),
        },
    }

    category_summary = []
    for cat in ("attitude", "acceleration", "angular_rate", "altitude", "motor", "command"):
        info = (flight.get("categories") or {}).get(cat)
        if not info:
            continue
        fam_events = [e for e in events if (e.get("family") or "") == cat]
        if not fam_events and not info.get("signals"):
            continue
        reviewed = [e for e in fam_events if e in attention_events]
        consistent_n = [e for e in fam_events if e.get("expected") and e.get("significance") == "physical"]
        if reviewed:
            text = f"{len(reviewed)} period{'s' if len(reviewed) != 1 else ''} not explained by the recorded input."
        elif consistent_n:
            n = len(consistent_n)
            text = (f"The {n} notable change{'s' if n != 1 else ''} "
                    f"{'follow' if n != 1 else 'follows'} the recorded input.")
        elif fam_events:
            text = "Only small or input-driven changes were recorded."
        else:
            text = "No notable change was detected."
        category_summary.append({"category": cat, "title": info.get("title", cat.title()), "text": text,
                                 "events": len(fam_events)})

    transitions = (flight.get("status_transitions") or [])
    label_of = {"armed": "arming status", "kill": "safety flag", "rc_failsafe": "RC failsafe flag",
                "tof_ok": "time-of-flight OK flag", "tof_status": "time-of-flight status",
                "block_flag": "block flag", "aux_1": "auxiliary signal"}
    status_changes = {
        "count": len(transitions),
        "signals": sorted({label_of.get(t.get("canonical") or "", t.get("canonical") or "") for t in transitions
                           if t.get("canonical")}),
        "note": ("Status flags changed during the recording. Their numeric encodings are not assumed, so they are "
                 "reported separately from flight behaviour."),
    }

    cannot = [
        "What the unverified signals physically represent: motor outputs, altitude and command inputs are "
        "reported as raw numbers because their units are not documented.",
        "Why the values changed.",
        "Whether any component was damaged, or whether the flight was safe.",
    ]
    if not overview["time_verified"]:
        cannot.insert(0, "Times in seconds; the file's time unit is not verified.")

    has_time = (timing.get("time_basis") or {}).get("status") not in (None, "none")
    if not has_time:
        time_note = "This file has no usable time column, so periods are given by row number."
    elif not _time_unit_verified(timing):
        time_note = "Times are approximate: the file does not record a verified time unit."
    else:
        time_note = "Times are seconds from the start of the recording."

    return {
        "mode": "ai_analysis",
        "mode_label": "AI Analysis",
        "mode_note": ("Read directly from this flight file and not scored by a trained model. "
                      "The interpretation compares commands, responses and related sensors."),
        "detection_note": "Deterministic telemetry analysis — no model score.",
        "summary": summary,
        "bottom_line": bottom_line,
        "time_note": time_note,
        "recording_s": (timing.get("duration_s") or {}).get("value"),
        "overview": overview,
        "system_checks": system_checks,
        "category_summary": category_summary,
        "status_changes": status_changes,
        "attention": [
            {
                "event_id": e.get("id"),
                "review_label": review_label(e.get("priority")),
                "priority": e.get("priority"),
                "title": e.get("headline") or _attention_title(e),
                "when": _when_direct(e, timing),
                "what_happened": _what_happened(e),
                "interpretation": e.get("interpretation"),
                "interpretation_label": e.get("interpretation_label"),
                "confidence": e.get("confidence"),
                "physical_values": _unique([physical_summary(p) for p in (e.get("physical_values") or [])
                                            if p.get("category") in aircraft_families])[:6],
                "command_evidence": _unique([command_sentence(c) for c in (e.get("command_checks") or [])])[:3],
                "consistency_evidence": _unique([consistency_sentence(c) for c in (e.get("consistency") or [])])[:3],
                "coincident": [o.get("headline") for o in events
                               if o is not e and o.get("category") != "state_transition"
                               and o.get("significance") == "physical"
                               and (o.get("family") or "") in aircraft_families
                               and o.get("start_index", 10**9) <= e.get("end_index", -1)
                               and e.get("start_index", 10**9) <= o.get("end_index", -1)][:3],
            }
            for e in attention_events
        ],
        "consistent_events": [
            {"event_id": e.get("id"), "title": e.get("headline") or _attention_title(e),
             "when": _when_direct(e, timing)["display"],
             "interpretation_label": e.get("interpretation_label")}
            for e in consistent
        ],
        "key_observations": observations,
        "next_steps": _next_steps(attention_events, timing),
        "limitation": LIMITATION,
        "other_intervals": [
            {
                "event_id": e.get("id"),
                "review_label": review_label(e.get("priority")),
                "title": e.get("headline") or _attention_title(e),
                "when": _when_direct(e, timing)["display"],
                "start_rel_s": _when_direct(e, timing).get("start_rel_s"),
                "end_rel_s": _when_direct(e, timing).get("end_rel_s"),
            }
            for e in other
        ],
        "what_system_cannot_determine": cannot,
    }


def _time_clusters(events: List[Dict[str, Any]]) -> List[Dict[str, int]]:
    clusters: List[Dict[str, int]] = []
    for e in sorted(events, key=lambda x: x.get("start_index", 0)):
        s, en = int(e.get("start_index", 0)), int(e.get("end_index", 0))
        if clusters and s <= clusters[-1]["end"]:
            clusters[-1]["end"] = max(clusters[-1]["end"], en)
        else:
            clusters.append({"start": s, "end": en})
    return clusters


def _attention_title(ev: Dict[str, Any]) -> str:
    if ev.get("headline"):
        return str(ev["headline"])
    cat = ev.get("category")
    if cat == "data_quality":
        if _has_kind(ev, "sampling_gap"):
            return "Gap in the recording"
        if _has_kind(ev, "missing_values"):
            return "Missing values in the recording"
        if _has_kind(ev, "flatline"):
            return "A value stopped changing"
        return "Problem in the recording"
    if cat == "state_transition":
        return "Stick, switch or status change"
    groups = _changed_groups(ev)
    if groups:
        names = [TITLE_NOUN.get(a.get("group"), GROUP_NAMES.get(a.get("group"), GROUP_NAMES["unclassified"])[0]) for a in groups[:2]]
        return " and ".join(names).capitalize() + " changed together"
    n = int(ev.get("n_channels_involved") or len(ev.get("channels") or []) or 1)
    return f"{_word(n).capitalize()} telemetry reading{'s' if n != 1 else ''} changed"


def _unique(items: List[str]) -> List[str]:
    return list(dict.fromkeys(i for i in items if i))


def _pilot_event_direct(ev: Dict[str, Any], timing: Dict[str, Any]) -> Dict[str, Any]:
    when = _when_direct(ev, timing)
    return {
        "event_id": ev.get("id"),
        "priority": ev.get("priority"),
        "review_label": review_label(ev.get("priority")),
        "category": ev.get("category"),
        "title": ev.get("headline") or _attention_title(ev),
        "what_happened": _what_happened(ev),
        "when": when,
        "why_review": _why_review(ev),
        "what_to_check": _what_to_check(ev, when),
        "interpretation": ev.get("interpretation"),
        "interpretation_label": ev.get("interpretation_label"),
        "confidence": ev.get("confidence"),
        "expected": bool(ev.get("expected")),
        "significance": ev.get("significance"),
        "physical_values": _unique([physical_summary(p) for p in (ev.get("physical_values") or [])
                                    if p.get("category") in {"attitude", "acceleration", "angular_rate", "altitude", "distance", "motor"}]),
        "command_evidence": _unique([command_sentence(c) for c in (ev.get("command_checks") or [])]),
        "consistency_evidence": _unique([consistency_sentence(c) for c in (ev.get("consistency") or [])]),
        "source_rows": {"start": ev.get("start_row"), "end": ev.get("end_row")},
        "technical_why": ev.get("why_selected", ""),
    }


def direct_event_cards(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    timing = data.get("timing") or {}
    return {e["id"]: _pilot_event_direct(e, timing) for e in data.get("events") or []}


# ---- Deep Analysis -------------------------------------------------------------------------------------------------


def deep_review_priority(ev: Any, n_signals: int) -> str:
    """Presentation order only: more affected signals first. Not a severity."""
    flagged = int(getattr(ev, "flagged_window_count", 0) or 0)
    if n_signals >= 3 or flagged >= 3:
        return "review_first"
    return "review"


def _deep_channel_ratios(detail: Dict[str, Any]) -> List[float]:
    out: List[float] = []
    for c in detail.get("channels") or []:
        before = (c.get("before") or {}).get("median")
        during = (c.get("during") or {}).get("mean")
        r = _ratio(before, during)
        if r is not None:
            out.append(r)
    return out


def _deep_what_happened(detail: Dict[str, Any]) -> str:
    labels = [str(c.get("pilot_label") or c.get("label") or "") for c in (detail.get("channels") or [])[:3] if c.get("pilot_label") or c.get("label")]
    checks = {str(c.get("kind")) for c in detail.get("signal_checks") or []}
    if "flatline" in checks:
        return "A recorded signal stopped changing and held the same value here, although it normally varies."
    if not labels:
        return "The recorded telemetry moved away from the earlier pattern in this period."
    subject = _join_items(labels)
    mag = _magnitude_text(_deep_channel_ratios(detail))
    tail = f" ({mag})" if mag else ""
    sentence = f"{_cap(subject)} moved away from the earlier pattern in this period{tail}."
    if "sudden_change" in checks:
        sentence += " One value changed abruptly."
    if "sustained_deviation" in checks:
        sentence += " At least one value stayed at a different level."
    return sentence


def _deep_why_review(detail: Dict[str, Any]) -> str:
    base = (
        "The trained check found this period unlike the rest of the recording "
        "and unlike the flights it learned from."
    )
    checks = {str(c.get("kind")) for c in detail.get("signal_checks") or []}
    if "sudden_change" in checks:
        base += " A separate check also found a sudden step in one recorded value."
    elif "sustained_deviation" in checks:
        base += " A separate check also found a value that stayed away from its earlier level."
    elif "flatline" in checks:
        base += " A separate check also found a value that stopped changing."
    base += " It does not show why the readings changed."
    return base


def _pilot_event_deep(detail: Dict[str, Any], priority: str) -> Dict[str, Any]:
    when = _when_deep(detail)
    where = when["display"].split(" into the flight")[0] if detail.get("start_label") else "this period"
    return {
        "event_id": detail.get("event_id"),
        "priority": priority,
        "review_label": review_label(priority),
        "category": "detector_event",
        "title": _deep_title(detail),
        "what_happened": _deep_what_happened(detail),
        "when": when,
        "why_review": _deep_why_review(detail),
        "what_to_check": [
            f"Replay the flight around {where} and note what the aircraft was doing.",
            "Compare it with what the pilot was doing and whether it was expected for that part of the flight.",
        ],
        "source_rows": {"start": None, "end": None},
        "technical_why": detail.get("what_observed", ""),
    }


def _deep_title(detail: Dict[str, Any]) -> str:
    labels = [str(c.get("pilot_label") or c.get("label") or "") for c in (detail.get("channels") or [])[:2] if c.get("pilot_label") or c.get("label")]
    if not labels:
        return "Telemetry changed during this period"
    if len(labels) == 1:
        return f"Change in {labels[0]}"
    return f"Changes in {labels[0]} and {labels[1]}"


def build_deep_pilot_report(
    *,
    report: Any,
    details: List[Dict[str, Any]],
    coverage: Optional[Dict[str, Any]],
    verdict: Dict[str, Any],
    analysed_s: Optional[float],
    flight_s: Optional[float],
) -> Dict[str, Any]:
    """Pilot briefing for *Deep Analysis* (deployed detector output)."""
    scored = []
    for d in details:
        n = len(d.get("channels") or [])
        scored.append((deep_review_priority(_DummyEvent(d), n), d))
    ranked_details = [d for _p, d in sorted(scored, key=lambda x: (REVIEW_ORDER[x[0]], -len(x[1].get("channels") or [])))]
    attention_details = ranked_details[:3]
    other_details = ranked_details[3:]

    if not ranked_details:
        summary = (
            "No period stood out in the part of the flight the check could see. "
            "That does not confirm that the aircraft was safe or that nothing changed."
        )
        bottom_line = (
            "Nothing stood out in the part of the flight that could be checked. "
            "That is not a clean bill of health."
        )
    else:
        lead = ranked_details[0]
        labels = [str(c.get("pilot_label") or c.get("label") or "") for c in (lead.get("channels") or [])[:3] if c.get("pilot_label") or c.get("label")]
        what = _join_items(labels) if labels else "the recorded telemetry"
        n = len(ranked_details)
        summary = (
            (f"The recording is about {flight_s:.0f} s long. " if flight_s else "")
            + f"{_word(n).capitalize()} period{'s' if n != 1 else ''} {'are' if n != 1 else 'is'} worth checking. "
            f"The first is at {lead.get('start_label')}–{lead.get('end_label')}, where {what} moved away from the earlier pattern. "
            "These are automatic findings, not confirmed faults."
        )
        bottom_line = "Start with the first period. These are automatic findings, not a diagnosis."

    observations: List[str] = []
    for d in attention_details:
        sentence = _deep_what_happened(d)
        if sentence not in observations:
            observations.append(sentence)
    if not observations:
        observations.append("The check found no notable changes to describe.")
    observations.append(
        "Only rotation, movement and attitude readings were checked. "
        "Height, pressure, motor outputs and radio inputs were not."
    )
    observations = observations[:5]

    steps: List[str] = []
    for d in attention_details[:3]:
        steps.append(f"Replay the flight around {d.get('start_label')}–{d.get('end_label')} and note what the aircraft was doing.")
    steps.append("Compare these changes with what the pilot was doing at the time.")
    steps.append("Check whether the behaviour was expected for that part of the flight.")
    steps.append("Open the original flight log if you need the exact values.")
    if coverage and coverage.get("status") not in (None, "ok"):
        steps.append("Treat the periods that could not be checked as unknown, not as normal.")
    next_steps = [{"event_id": d.get("event_id", ""), "text": t} for d, t in zip(attention_details[:3], steps[:3])]
    next_steps += [{"event_id": "", "text": t} for t in steps[3 if attention_details else 0:]]

    return {
        "mode": "deep_analysis",
        "mode_label": "Deep Analysis",
        "mode_note": (
            "This report uses the deployed anomaly-detection model. It shows where the recorded signals "
            "differed from the patterns the model learned; it does not confirm a fault."
        ),
        "detection_note": "Deployed anomaly model — automatic findings, not confirmed faults.",
        "summary": summary,
        "bottom_line": bottom_line,
        "time_note": "Times are mm:ss from the start of the recording.",
        "recording_s": flight_s,
        "attention": [
            {
                "event_id": d.get("event_id"),
                "review_label": review_label(deep_review_priority(_DummyEvent(d), len(d.get("channels") or []))),
                "priority": deep_review_priority(_DummyEvent(d), len(d.get("channels") or [])),
                "title": _deep_title(d),
                "when": _when_deep(d),
                "what_happened": _deep_what_happened(d),
            }
            for d in attention_details
        ],
        "key_observations": observations,
        "next_steps": next_steps,
        "limitation": LIMITATION,
        "other_intervals": [
            {
                "event_id": d.get("event_id"),
                "review_label": review_label(deep_review_priority(_DummyEvent(d), len(d.get("channels") or []))),
                "title": _deep_title(d),
                "when": _when_deep(d)["display"],
                "start_rel_s": _when_deep(d).get("start_rel_s"),
                "end_rel_s": _when_deep(d).get("end_rel_s"),
            }
            for d in other_details
        ],
        "what_system_cannot_determine": [
            "Why the readings changed.",
            "Whether any component was damaged, or whether the flight was safe.",
            "What happened during periods the check could not cover.",
        ],
        "coverage": {
            "analysed_s": analysed_s,
            "flight_s": flight_s,
            "status": (coverage or {}).get("status"),
            "analysed_fraction": (coverage or {}).get("analysed_fraction"),
        },
    }


def deep_event_cards(details: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for d in details:
        p = deep_review_priority(_DummyEvent(d), len(d.get("channels") or []))
        out[str(d.get("event_id"))] = _pilot_event_deep(d, p)
    return out


class _DummyEvent:
    """Adapter so the priority helper can use the same fields in both modes."""

    def __init__(self, detail: Dict[str, Any]):
        self.flagged_window_count = int(detail.get("flagged_window_count") or 0)
