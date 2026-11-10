"""Deterministic operator-facing report for *Direct AI Telemetry Analysis*.

Built only from the computed summary (events, measurements, associations, timing); no model involved, so it is
always available, including when the AI step fails. It answers, per event: what changed, when, which signals,
relative to what earlier behaviour, what else changed at about the same time, what can and cannot be concluded,
and what to review next.

Wording rules (enforced by tests): no fault diagnosis, no safety judgement, no causation, no physical units or
directions for unverified channels, "higher/lower raw values" instead of motion words, and time in seconds only
as an explicitly conditional conversion unless the time unit is verified.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.event_engine import BASELINE_MIN
from src.pilot_report import build_direct_pilot_report, direct_event_cards

GROUP_LABEL = {
    "motor_output": "motor-labelled", "gyroscope": "gyroscope-labelled", "accelerometer": "accelerometer-labelled",
    "attitude": "attitude-labelled", "distance_height": "distance/height-labelled", "controller_internal": "controller-internal",
    "pilot_command": "pilot-command", "status_flag": "status-flag",
}
_NUMWORD = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
UNAVAILABLE = "Unavailable"


def fmt(v: Any) -> str:
    """Compact, honest number formatting; None becomes 'Unavailable' (never an empty cell or a zero)."""
    if v is None:
        return UNAVAILABLE
    if isinstance(v, bool):
        return str(v)
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if x == 0:
        return "0"
    if abs(x) >= 100:
        return f"{x:.0f}"
    if abs(x) >= 1:
        return f"{x:.2f}".rstrip("0").rstrip(".")
    return f"{x:.3g}"


def _n(k: int) -> str:
    return _NUMWORD.get(k, str(k))


def when_text(ev: Dict[str, Any], timing: Dict[str, Any]) -> str:
    rows = f"rows {ev['start_row']}–{ev['end_row']}"
    if ev.get("start_raw") is None:
        return rows + " (no time column)"
    col = timing.get("time_column") or "time"
    t = f"{col} {fmt(ev['start_raw'])}–{fmt(ev['end_raw'])} (raw ticks)"
    sec = ev.get("span_s_assumed")
    if sec and sec.get("value") is not None:
        basis = sec["basis"]
        t += f", about {fmt(sec['value'])} s" + ("" if basis == "verified" else f" ({basis.lower()}; unit not verified)")
    return f"{rows} · {t}"


def _sem_phrase(m: Dict[str, Any]) -> str:
    g = GROUP_LABEL.get(m.get("group") or "", None)
    if m.get("role") == "command":
        return "pilot-command input"
    if g:
        return f"{g} channel" + (", meaning unverified" if "unverified" in (m.get("semantics") or "") else "")
    return "raw channel, meaning unverified"


def when_short(ev: Dict[str, Any], timing: Dict[str, Any]) -> str:
    rows = f"rows {ev['start_row']}–{ev['end_row']}"
    if ev.get("start_raw") is None:
        return rows
    return f"{rows} ({timing.get('time_column') or 'time'} {fmt(ev['start_raw'])}–{fmt(ev['end_raw'])} raw ticks)"


def _times(x: Any) -> str:
    return "over 20×" if abs(x) > 20 else f"{fmt(abs(x))}×"


def channel_line(m: Dict[str, Any]) -> str:
    """One measured sentence about one channel, relative to its own earlier behaviour."""
    b, d, a = m["before"], m["during"], m["after"]
    if "status_change" in m["kinds"]:
        if b["available"] and a["available"] and b["median"] != a["median"]:
            return f"{m['channel']} ({_sem_phrase(m)}): value {fmt(b['median'])} before, {fmt(a['median'])} after; transition at row {m['onset_row']}."
        note = "" if b["available"] else f" No earlier baseline is available (fewer than {BASELINE_MIN} earlier samples)."
        return f"{m['channel']} ({_sem_phrase(m)}): changed value at row {m['onset_row']} (range {fmt(d['min'])} to {fmt(d['max'])}).{note}"
    if b["available"]:
        core = (f"{m['channel']} ({_sem_phrase(m)}): median {fmt(b['median'])} before, {fmt(d['median'])} during "
                f"(range {fmt(d['min'])} to {fmt(d['max'])})")
    else:
        core = (f"{m['channel']} ({_sem_phrase(m)}): no earlier baseline is available (fewer than {BASELINE_MIN} earlier samples); "
                f"during the interval it ranged {fmt(d['min'])} to {fmt(d['max'])}")
    core += f", {fmt(a['median'])} after" if a["available"] else ", post-event comparison unavailable"
    if m.get("shift_sigma") is not None and abs(m["shift_sigma"]) >= 1:
        core += f"; the median moved {_times(m['shift_sigma'])} the channel's earlier spread"
    if "variability_increase" in m["kinds"] and b["available"] and b.get("jitter") is not None and d.get("jitter") is not None:
        core += f"; sample-to-sample variation (robust spread) {fmt(b['jitter'])} before, {fmt(d['jitter'])} during"
    if m.get("persistence"):
        core += f"; {m['persistence']}"
    return core + "."


def _group_statement(a: Dict[str, Any], by_ch: Dict[str, Dict[str, Any]]) -> str:
    label = GROUP_LABEL.get(a["group"], a["group"])
    k, total = len(a["channels"]), a["n_in_file"]
    chans = ", ".join(sorted(a["channels"]))
    who = (f"Both {label} channels ({chans})" if k == total == 2 else f"All {_n(k)} {label} channels ({chans})") if k == total else f"{_n(k)} of {_n(total)} {label} channels ({chans})"
    mv = {"higher": "had higher raw values", "lower": "had lower raw values", "mixed": "moved in different raw directions"}[a["direction"]]
    return f"{who} {mv} during the interval, relative to their earlier medians."


def association_statements(ev: Dict[str, Any]) -> List[Dict[str, str]]:
    """Plain statements of observed associations, each labelled with what kind of evidence supports it."""
    by_ch = {m["channel"]: m for m in ev["measurements"]}
    out: List[Dict[str, str]] = []
    for a in ev.get("associations", []):
        if a["type"] == "group":
            out.append({"kind": "co-movement", "text": _group_statement(a, by_ch)})
        elif a["type"] == "simultaneous":
            sp = a["onset_spread_samples"]
            out.append({"kind": "simultaneous", "text": f"{', '.join(a['channels'])} began changing within {sp} sample{'s' if sp != 1 else ''} of each other, around row {a['first_row']}."})
        elif a["type"] == "sequential":
            steps = a["steps"]
            if len(steps) > 6:
                steps = steps[:6]
            seq = "; then ".join(f"{', '.join(s['channels'])} (row {s['onset_row']})" for s in steps)
            out.append({"kind": "sequential", "text": f"Changes began in this order: {seq}. The order is a timing observation only."})
        elif a["type"] == "correlation":
            out.append({"kind": "statistical correlation", "text": f"{a['a']} and {a['b']} are correlated within this window (r = {a['r']:.2f}, {a['n']} samples). Correlation does not establish a physical relationship."})
    return out


def event_title(ev: Dict[str, Any]) -> str:
    cat, ms = ev["category"], ev["measurements"]
    if cat == "data_quality":
        if "sampling_gap" in ev["kinds"]:
            return "Gap in the time column"
        what = {"missing_values": "Missing values", "flatline": "Repeated identical values"}
        k = next((x for x in ev["kinds"] if x in what), "Data-quality issue")
        return f"{what.get(k, k)} in {', '.join(ev['channels'][:3])}" if ev["channels"] else what.get(k, k)
    if cat == "state_transition":
        parts = []
        for m in ms[:3]:
            if "status_change" in m["kinds"]:
                ch = m["before_median"] is not None and m["after_median"] is not None and m["before_median"] != m["after_median"]
                parts.append(f"{m['channel']} {fmt(m['before_median'])} → {fmt(m['after_median'])}" if ch else f"{m['channel']} changed")
            else:
                parts.append(f"{m['channel']} moved")
        return "Status / pilot-input change: " + ", ".join(parts)
    groups: Dict[str, int] = {}
    for m in ms:
        g = m.get("group")
        if g and g not in ("unclassified", "status_flag"):
            groups[g] = groups.get(g, 0) + 1
    if len(ms) == 1 or not groups:
        names = ", ".join(m["channel"] for m in ms[:3])
        return f"Change in {names}" + (f" and {len(ms) - 3} more" if len(ms) > 3 else "")
    ordered = sorted(groups, key=lambda g: -groups[g])[:3]
    return "Changes in " + ", ".join(GROUP_LABEL.get(g, g) + " channels" for g in ordered) + (" and others" if len(groups) > 3 or len(ms) > sum(groups.values()) else "")


def event_narrative(ev: Dict[str, Any], timing: Dict[str, Any]) -> Dict[str, Any]:
    ms = ev["measurements"]
    top = ms[:6]
    unknown: List[str] = []
    if any("unverified" in (m.get("semantics") or "") for m in ms):
        unknown.append("What the unverified-meaning channels represent physically.")
    unknown.append("Why the values changed: the file does not record the reason.")
    if any(not m["before"]["available"] for m in ms):
        unknown.append("Behaviour before the interval for channels with no baseline (too few earlier samples).")
    if any(not m["after"]["available"] for m in ms):
        unknown.append("Whether the change persisted, for channels with no post-event data.")
    if timing.get("time_basis", {}).get("status") != "verified" and ev.get("start_raw") is not None:
        unknown.append("The time unit is not verified, so durations in seconds are conditional.")
    stmts = association_statements(ev)
    concl: List[str] = []
    if ev["category"] == "statistical_irregularity" and ms:
        concl.append(f"Raw values of {len(ev['channels'])} channel{'s' if len(ev['channels']) != 1 else ''} departed from their own earlier behaviour in this interval.")
    elif ev["category"] == "state_transition":
        concl.append("A status or pilot-command channel changed value. Command channels are inputs, so a change there can be expected.")
    else:
        concl.append("The data in this interval is missing, repeated or irregularly timed, so it cannot support conclusions about the aircraft.")
    if any(s["kind"] in ("simultaneous", "co-movement") for s in stmts):
        concl.append("Several channels changed at about the same time; this is a timing association, not an established physical relationship.")
    concl.append("This analysis makes no statement about flight safety or the condition of any component.")
    review = [f"Compare {when_short(ev, timing)} with pilot notes, video or the original log."]
    if ev["concurrent_with"]:
        review.append("Review together with " + ", ".join(f"{c['event_id']} ({c['relation']})" for c in ev["concurrent_with"]) + ".")
    if any(m.get("role") == "command" for m in ms):
        review.append("Check the pilot's control inputs for this period.")
    return {
        "title": event_title(ev),
        "when": when_text(ev, timing),
        "what_changed": [channel_line(m) for m in top] + ([f"{len(ms) - len(top)} further channel(s) are listed in the measurements table."] if len(ms) > len(top) else []),
        "same_time": stmts,
        "can_conclude": concl,
        "unknown": unknown,
        "review_next": review,
        "why_selected": ev.get("why_selected", ""),
    }


def build_operator_report(data: Dict[str, Any]) -> Dict[str, Any]:
    timing, events = data["timing"], data["events"]
    dq = data["data_quality"]
    tb = timing.get("time_basis", {})
    narratives = {ev["id"]: event_narrative(ev, timing) for ev in events}
    first = [e for e in events if e.get("priority") == "review_first"]
    irregular = [e for e in events if e["category"] == "statistical_irregularity"]
    bits = [f"{dq['rows']} rows, {dq['numeric_columns']} numeric columns"]
    if timing.get("span_raw") is not None:
        bits.append(f"time column {timing['time_column']} spans {fmt(timing['span_raw'])} raw ticks"
                    + (f" (about {fmt(timing['duration_s']['value'])} s, {timing['duration_s']['basis'].lower()})" if timing.get("duration_s") else ""))
    flight = ["Recording: " + "; ".join(bits) + "."]
    ph = data.get("phases") or []
    if len(ph) >= 2:
        flight.append(f"Status flag '{ph[0]['flag']}' changes from {fmt(ph[0]['value'])} to {fmt(ph[1]['value'])} at rows {ph[0]['end_row']}–{ph[1]['start_row']}.")
    if not events:
        headline = "The statistical screen found no interval in which the raw channels departed from their earlier behaviour. That does not show the recording is normal."
    else:
        pool = first or irregular or events
        lead = max(pool, key=lambda e: (e.get("n_channels_involved", 0), -e["start_index"]))
        n = narratives[lead["id"]]
        groups = [a for a in lead.get("associations", []) if a["type"] == "group"]
        order = ["motor_output", "gyroscope", "attitude", "accelerometer", "distance_height", "controller_internal", "pilot_command"]
        groups.sort(key=lambda a: (-(len(a["channels"]) == a["n_in_file"]), order.index(a["group"]) if a["group"] in order else 99))
        what = ("; ".join(_group_statement(a, {}).replace(", relative to their earlier medians", "").rstrip(".") for a in groups[:2]) if groups
                else n["title"][0].lower() + n["title"][1:])
        n_ch = lead.get("n_channels_involved", 0)
        headline = (f"{len(events)} interval{'s' if len(events) != 1 else ''} to review. The most extensive starts at row {lead['start_row']}: "
                    f"{what}" + (f", with {n_ch} channels in total departing from their earlier behaviour" if n_ch > 3 else "") + ".")
    recs: List[Dict[str, str]] = []
    for e in (first or events)[:4]:
        recs.append({"event_id": e["id"], "action": f"Compare {when_short(e, timing)} with pilot notes, video or the original log.",
                     "why": "Raw channels changed together there; the file alone cannot say why."})
    if any(e["category"] == "state_transition" for e in events):
        recs.append({"event_id": "", "action": "Match the status and pilot-command changes against the pilot's actual inputs.",
                     "why": "Command channels are inputs; changes there may be intended."})
    material: List[Dict[str, str]] = []
    if tb.get("status") != "verified":
        material.append({"issue": "Time unit not verified", "detail": f"Time is shown in raw ticks of '{timing.get('time_column')}'. "
                         f"Seconds are only a conditional conversion (hypothesis: {tb.get('unit_hypothesis')})."})
    if timing.get("n_gaps") or timing.get("out_of_order_steps") or timing.get("rows_without_valid_timestamp") or timing.get("repeated_timestamps"):
        material.append({"issue": "Timestamp irregularities", "detail": f"{timing.get('n_gaps', 0)} gap(s), {timing.get('repeated_timestamps', 0)} repeated, "
                         f"{timing.get('out_of_order_steps', 0)} out of order, {timing.get('rows_without_valid_timestamp', 0)} invalid."})
    if dq["channels_with_missing"]:
        material.append({"issue": "Missing values", "detail": ", ".join(f"{k} ({v['missing'] + v['non_finite']})" for k, v in dq["channels_with_missing"].items())})
    unv = sorted({c for e in events for c in e["channels"] if (data["channels"].get(c) or {}).get("meaning_status") != "verified"})
    if unv:
        material.append({"issue": "Unverified channel meanings", "detail": "Channels in the events above are raw columns with unverified meanings: " + ", ".join(unv[:12]) + "."})
    if any(not m["before"]["available"] or not m["after"]["available"] for e in events for m in e["measurements"]):
        material.append({"issue": "Incomplete comparison windows", "detail": "Some events are too close to the start or end of the recording for a full before/after comparison; those cells show 'Unavailable'."})
    return {
        "headline": headline, "flight_summary": flight,
        "events": narratives,
        "recommended_review": recs, "data_quality_material": material,
        "pilot_report": build_direct_pilot_report(data),
        "pilot_events": direct_event_cards(data),
        "essential_limitations": [
            "This is a statistical description of raw columns. It is not validated by the deployed anomaly detector.",
            "It does not establish a defect, a cause, or a safety judgement.",
        ],
        "not_established": "Associations between channels are observed timing and statistical correlation; no physical relationship is claimed.",
    }
