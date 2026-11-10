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

LIMITATION = (
    "Some telemetry signals do not have verified physical meanings or units. "
    "Their changes can be observed, but their physical significance cannot yet be established."
)

REVIEW_LABEL = {
    "review_first": "Review recommended",
    "review": "Worth checking",
    "minor": "No notable change detected",
}

REVIEW_ORDER = {"review_first": 0, "review": 1, "minor": 2}

GROUP_NAMES = {
    "motor_output": ("motor reading", "motor readings"),
    "gyroscope": ("rotation reading", "rotation readings"),
    "accelerometer": ("movement reading", "movement readings"),
    "attitude": ("attitude reading", "attitude readings"),
    "distance_height": ("height/distance reading", "height/distance readings"),
    "controller_internal": ("controller reading", "controller readings"),
    "pilot_command": ("pilot input", "pilot inputs"),
    "status_flag": ("status reading", "status readings"),
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


def _group_sentence(a: Dict[str, Any]) -> str:
    count = len(a.get("channels") or [])
    total = int(a.get("n_in_file") or count)
    who = group_phrase(a.get("group"), count, total)
    direction = a.get("direction")
    if direction == "higher":
        verb = "was higher" if count == 1 else "were higher"
        return f"{who} {verb} than earlier in the recording."
    if direction == "lower":
        verb = "was lower" if count == 1 else "were lower"
        return f"{who} {verb} than earlier in the recording."
    verb = "changed" if count == 1 else "changed"
    return f"{who} {verb} at about the same time, in different directions."


def _measurement_group(m: Dict[str, Any]) -> str:
    return str(m.get("group") or ("status_flag" if m.get("role") == "status" else "unclassified"))


def _singular_group(group: str) -> str:
    return GROUP_NAMES.get(group, GROUP_NAMES["unclassified"])[0]


def _event_group_sentences(ev: Dict[str, Any], limit: int = 3) -> List[str]:
    groups = _changed_groups(ev)
    out = [_group_sentence(a) for a in groups[:limit]]
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
        if m.get("direction") == "higher":
            out.append(f"The {label} was higher than earlier in the recording.")
        elif m.get("direction") == "lower":
            out.append(f"The {label} was lower than earlier in the recording.")
        else:
            out.append(f"The {label} moved away from its earlier pattern.")
        return out
    if len(seen) == 1:
        n = len(measurements)
        label = GROUP_NAMES.get(seen[0], GROUP_NAMES["unclassified"])[1]
        directions = {str(m.get("direction") or "") for m in measurements}
        if directions == {"higher"}:
            out.append(f"{_word(n).capitalize()} {label} were higher than earlier in the recording.")
        elif directions == {"lower"}:
            out.append(f"{_word(n).capitalize()} {label} were lower than earlier in the recording.")
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
                parts.append("a pilot input")
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
        if _sustained(ev):
            text += " The different levels were still present later in the recording."
        if _simultaneous(ev):
            text += " These changes began close together."
        return text

    n = int(ev.get("n_channels_involved") or len(ev.get("channels") or []) or 1)
    noun = "reading" if n == 1 else "readings"
    return f"{_word(n).capitalize()} telemetry {noun} moved away from the earlier pattern in this interval."


def _why_review(ev: Dict[str, Any], technical_why: Optional[str] = None) -> str:
    cat = ev.get("category")
    priority = ev.get("priority")
    if cat == "data_quality":
        base = "The recording itself is incomplete or irregular here, which limits what can be concluded about this period."
    elif cat == "state_transition":
        base = "A pilot input or recorded status changed here, so it is worth matching against what was intended."
    elif priority == "review_first":
        base = "More different readings changed here than in any other period."
    elif priority == "review":
        base = "Several readings changed away from their earlier pattern in this interval."
    else:
        base = "This interval was recorded for completeness; the changes were limited."
    if _sustained(ev):
        base += " The new level did not immediately return to the earlier pattern."
    if _simultaneous(ev) and cat != "data_quality":
        base += " The changes happened close together, but the file cannot show that one caused another."
    return base


def _when_direct(ev: Dict[str, Any], timing: Dict[str, Any]) -> Dict[str, Any]:
    rows = _rows(ev)
    start_raw, end_raw = ev.get("start_raw"), ev.get("end_raw")
    if start_raw is None or end_raw is None:
        return {
            "display": f"{rows} (the file has no usable time column)",
            "approximate": "",
            "start_row": ev.get("start_row"),
            "end_row": ev.get("end_row"),
        }
    display = f"{rows} · raw timestamps {_int(start_raw)}–{_int(end_raw)}"
    approx = ""
    span = ev.get("span_s_assumed") or {}
    if span.get("value") is not None:
        basis = str(span.get("basis") or "")
        if basis == "verified":
            first = timing.get("first_raw")
            if first is not None:
                approx = f"About {_elapsed(float(start_raw) - float(first))}–{_elapsed(float(end_raw) - float(first))} into the recording."
            else:
                approx = f"About {float(span['value']):.1f} s long."
        else:
            unit = str(timing.get("time_basis", {}).get("unit_hypothesis") or "").lower()
            scale = {"ms": 1e-3, "s": 1.0, "us": 1e-6, "ns": 1e-9}.get(unit)
            first = timing.get("first_raw")
            if scale and first is not None:
                a = (float(start_raw) - float(first)) * scale
                b = (float(end_raw) - float(first)) * scale
                approx = (
                    f"Approximately {_elapsed(a)}–{_elapsed(b)} into the recording if the time column is in "
                    f"{'milliseconds' if unit == 'ms' else unit} (the time unit is not verified)."
                )
    return {
        "display": display,
        "approximate": approx,
        "start_row": ev.get("start_row"),
        "end_row": ev.get("end_row"),
        "start_raw": start_raw,
        "end_raw": end_raw,
    }


def _when_deep(detail: Dict[str, Any]) -> Dict[str, Any]:
    start, end = detail.get("start_label"), detail.get("end_label")
    duration = detail.get("duration_s")
    display = f"{start}–{end} into the recording" + (f" (about {duration:.1f} s)" if duration is not None else "")
    segment = detail.get("session_id")
    if segment is not None:
        display += f", flight segment {int(segment) + 1}"
    return {
        "display": display,
        "approximate": "",
        "start_row": None,
        "end_row": None,
        "start_rel_s": detail.get("start_rel_s"),
        "end_rel_s": detail.get("end_rel_s"),
    }


def _what_to_check(ev: Dict[str, Any], when: Dict[str, Any]) -> List[str]:
    where = f"rows {when['start_row']}–{when['end_row']}" if when.get("start_row") is not None else when.get("display", "this period")
    cat = ev.get("category")
    if cat == "data_quality":
        return [
            f"Check the flight recording at {where} and see whether the aircraft was still being flown through the gap.",
            "If the missing period matters, compare it with the original flight log or download.",
        ]
    if cat == "state_transition":
        return [
            f"Compare the change at {where} with the pilot's inputs and what was intended for that part of the flight.",
            "Check whether the status change was expected at that point.",
        ]
    checks = [
        f"Review the flight recording around {where} and note what the aircraft was doing.",
        "Compare the event with the pilot inputs and check whether it was expected for that flight phase.",
    ]
    if _has_kind(ev, "flatline"):
        checks[1] = "Review the original flight-controller log to see whether the values were recorded correctly."
    return checks


def _observation_sentences(ev: Dict[str, Any], timing: Dict[str, Any]) -> List[str]:
    cat = ev.get("category")
    when = _when_direct(ev, timing)
    if cat == "data_quality":
        return [f"{when['display']}: part of the recording could not be assessed."]
    if cat == "state_transition":
        groups = []
        for m in ev.get("measurements", [])[:2]:
            g = m.get("group") or ("status_flag" if m.get("role") == "status" else "unclassified")
            groups.append(GROUP_NAMES.get(g, GROUP_NAMES["unclassified"])[0])
        uniq = list(dict.fromkeys(groups)) or ["recorded status"]
        what = " and ".join(uniq)
        return [f"{when['display']}: {what} changed. These are inputs or recorded states, not measured movement."]
    out = _event_group_sentences(ev, limit=3)
    if _sustained(ev) and out:
        out[-1] = out[-1].rstrip(".") + " and stayed at the new level."
    if not out:
        out = [f"{when['display']}: the recorded telemetry changed away from its earlier pattern."]
    return out


def _lead_summary_clause(ev: Dict[str, Any]) -> str:
    cat = ev.get("category")
    where = _rows(ev)
    if cat == "data_quality":
        return f"The main limitation is a data problem around {where}."
    if cat == "state_transition":
        return f"The main change is a pilot input or status change around {where}."
    groups = _changed_groups(ev)
    if groups:
        names = [GROUP_NAMES.get(a.get("group"), GROUP_NAMES["unclassified"])[1] for a in groups[:3]]
        if len(names) == 1:
            what = names[0]
        elif len(names) == 2:
            what = f"{names[0]} and {names[1]}"
        else:
            what = ", ".join(names[:-1]) + " and " + names[-1]
        return f"The largest change started around {where}, where {what} changed close together."
    return f"The largest change started around {where}."


def _duration_text(timing: Dict[str, Any], rows: int) -> str:
    dur = timing.get("duration_s") or {}
    if dur.get("value") is not None:
        value = float(dur["value"])
        basis = str(dur.get("basis") or "")
        if basis == "verified":
            return f"The recording covers about {value:.1f} s"
        unit = str(timing.get("time_basis", {}).get("unit_hypothesis") or "")
        label = "milliseconds" if unit == "ms" else (unit or "an unverified unit")
        return f"The recording covers about {value:.1f} s of telemetry if the time column is in {label} (the time unit is not verified)"
    return f"The recording has {_int(rows)} rows and no usable time column"


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
        where = f"rows {when['start_row']}–{when['end_row']}" if when.get("start_row") is not None else when.get("display", "this period")
        if ev.get("category") == "data_quality":
            add(f"Check the flight recording at {where} for missing data or a download problem.", ev.get("id", ""))
        elif ev.get("category") == "state_transition":
            add(f"Compare the input or status change at {where} with what was intended for that flight phase.", ev.get("id", ""))
        else:
            add(f"Review the flight recording around {where} and note what the aircraft was doing.", ev.get("id", ""))
    add("Compare the changes with the pilot inputs and the flight phase.")
    add("Check whether the behaviour was expected during that part of the flight.")
    add("Review the original flight-controller log if you need the exact values.")
    return out


def build_direct_pilot_report(data: Dict[str, Any]) -> Dict[str, Any]:
    """Pilot briefing for *AI Analysis* (deterministic direct telemetry analysis)."""
    timing = data.get("timing") or {}
    events = list(data.get("events") or [])
    ranked = sorted(events, key=lambda e: (REVIEW_ORDER.get(e.get("priority"), 1), e.get("start_index", 0)))
    attention_events = [e for e in ranked if e.get("priority") != "minor"][:3]
    if not attention_events:
        attention_events = ranked[:2]
    attention_ids = {e.get("id") for e in attention_events}
    other = [e for e in ranked if e.get("id") not in attention_ids][:8]

    if not events:
        summary = (
            "The system did not find any period of notable change in the part of the recording it could assess. "
            "That does not confirm that the flight was normal or that the aircraft was safe."
        )
    else:
        lead = ranked[0]
        n = len(events)
        n_attention = len(attention_events)
        if n_attention == n:
            review_clause = f"{n} period{'s' if n != 1 else ''} came up for review."
        else:
            review_clause = (
                f"{n_attention} period{'s' if n_attention != 1 else ''} "
                f"{'need' if n_attention != 1 else 'needs'} a closer look; "
                f"{n - n_attention} other interval{'s' if n - n_attention != 1 else ''} "
                f"{'are' if n - n_attention != 1 else 'is'} listed for reference."
            )
        summary = (
            f"{_duration_text(timing, int(data.get('data_quality', {}).get('rows') or 0))}. "
            f"{review_clause} "
            f"{_lead_summary_clause(lead)} "
            "The file shows what changed, but not why."
        )

    observations: List[str] = []
    for ev in attention_events[:3]:
        for sentence in _observation_sentences(ev, timing):
            if sentence not in observations:
                observations.append(sentence)
    if not observations:
        observations.append("The system found no notable changes to describe.")
    observations = observations[:5]

    return {
        "mode": "ai_analysis",
        "mode_label": "AI Analysis",
        "mode_note": (
            "These are computed telemetry observations from this file. "
            "They are not scored by the deployed anomaly detector."
        ),
        "detection_note": "Computed observations — no model score.",
        "summary": summary,
        "attention": [
            {
                "event_id": e.get("id"),
                "review_label": review_label(e.get("priority")),
                "priority": e.get("priority"),
                "title": _attention_title(e),
                "when": _when_direct(e, timing),
                "what_happened": _what_happened(e),
            }
            for e in attention_events
        ],
        "key_observations": observations,
        "next_steps": _next_steps(attention_events, timing),
        "limitation": LIMITATION,
        "other_intervals": [
            {
                "event_id": e.get("id"),
                "review_label": review_label(e.get("priority")),
                "title": _attention_title(e),
                "when": _when_direct(e, timing)["display"],
            }
            for e in other
        ],
        "what_system_cannot_determine": [
            "What the readings physically represent, where their meaning or unit is not verified.",
            "Why the readings changed.",
            "Whether a component was damaged or the aircraft was safe.",
        ],
    }


def _attention_title(ev: Dict[str, Any]) -> str:
    cat = ev.get("category")
    where = _rows(ev)
    if cat == "data_quality":
        if _has_kind(ev, "sampling_gap"):
            return f"Gap in the recording at {where}"
        if _has_kind(ev, "missing_values"):
            return f"Missing values at {where}"
        if _has_kind(ev, "flatline"):
            return f"Values stopped changing at {where}"
        return f"Recording problem at {where}"
    if cat == "state_transition":
        return f"Pilot input or status change at {where}"
    groups = _changed_groups(ev)
    if groups:
        names = [GROUP_NAMES.get(a.get("group"), GROUP_NAMES["unclassified"])[1] for a in groups[:2]]
        return " and ".join(names).capitalize() + " changed together"
    n = int(ev.get("n_channels_involved") or len(ev.get("channels") or []) or 1)
    return f"{_word(n).capitalize()} telemetry reading{'s' if n != 1 else ''} changed"


def _pilot_event_direct(ev: Dict[str, Any], timing: Dict[str, Any]) -> Dict[str, Any]:
    when = _when_direct(ev, timing)
    return {
        "event_id": ev.get("id"),
        "priority": ev.get("priority"),
        "review_label": review_label(ev.get("priority")),
        "category": ev.get("category"),
        "title": _attention_title(ev),
        "what_happened": _what_happened(ev),
        "when": when,
        "why_review": _why_review(ev),
        "what_to_check": _what_to_check(ev, when),
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


def _deep_what_happened(detail: Dict[str, Any]) -> str:
    labels = [str(c.get("pilot_label") or c.get("label") or "") for c in (detail.get("channels") or [])[:3] if c.get("pilot_label") or c.get("label")]
    checks = {str(c.get("kind")) for c in detail.get("signal_checks") or []}
    if "flatline" in checks:
        return "A recorded signal stopped changing and held the same value here, although it normally varies."
    if not labels:
        return "The recorded telemetry moved away from the earlier pattern in this period."
    subject = _join_items(labels)
    sentence = f"{_cap(subject)} moved away from the earlier pattern in this period."
    if "sudden_change" in checks:
        sentence += " One value changed abruptly."
    if "sustained_deviation" in checks:
        sentence += " At least one value stayed at a different level."
    return sentence


def _deep_why_review(detail: Dict[str, Any]) -> str:
    base = (
        "The automatic check marked this period as different from the rest of the recording "
        "and from the flight patterns it learned."
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
    where = f"{detail.get('start_label')}–{detail.get('end_label')}" if detail.get("start_label") else "this period"
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
            f"Review the flight recording around {where} and note what the aircraft was doing.",
            "Compare the event with the pilot inputs and check whether it was expected for that flight phase.",
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
            "The automatic check did not flag any period in the part of the flight it could assess. "
            "That does not confirm that the aircraft was safe or that nothing changed."
        )
    else:
        lead = ranked_details[0]
        labels = [str(c.get("pilot_label") or c.get("label") or "") for c in (lead.get("channels") or [])[:3] if c.get("pilot_label") or c.get("label")]
        what = _join_items(labels) if labels else "the recorded telemetry"
        n = len(ranked_details)
        summary = (
            f"The automatic check flagged {n} period{'s' if n != 1 else ''} for review. "
            f"The first runs from {lead.get('start_label')} to {lead.get('end_label')} into the recording, "
            f"where {what} moved away from the earlier pattern. "
            "These are automatic findings, not confirmed faults."
        )
        if flight_s:
            summary = (
                f"The recording is about {flight_s:.0f} s long. " + summary
            )

    observations: List[str] = []
    for d in attention_details:
        sentence = _deep_what_happened(d)
        if sentence not in observations:
            observations.append(sentence)
    if not observations:
        observations.append("The automatic check found no notable changes to describe.")
    observations.append(
        "Only the attitude and inertial signals used by the detector were checked; "
        "altitude, pressure, motor outputs and radio inputs were not."
    )
    observations = observations[:5]

    steps: List[str] = []
    for d in attention_details[:3]:
        steps.append(f"Review the flight recording around {d.get('start_label')}–{d.get('end_label')} and note what the aircraft was doing.")
    steps.append("Compare the changes with the pilot inputs and check whether they were expected for that flight phase.")
    steps.append("Review the original flight-controller log if you need the exact values.")
    if coverage and coverage.get("status") not in (None, "ok"):
        steps.append("Treat the periods that could not be assessed as unknown, not as normal.")
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
            }
            for d in other_details
        ],
        "what_system_cannot_determine": [
            "Why the readings changed.",
            "Whether a component was damaged or the aircraft was safe.",
            "What happened during periods the detector could not assess.",
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
