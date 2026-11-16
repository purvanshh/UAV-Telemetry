"""The agent loop: tool calls until the model submits its report, then validation. Never raises."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.agent import prompt
from src.agent.sources import Source
from src.agent.symptoms import SYMPTOMS
from src.agent.tools import Toolbox
from src.agent.validate import validate
from src.llm_client import LLMError, OpenAIChatClient, api_configured, get_model_name

log = logging.getLogger("uav.agent")

MAX_ROUNDS = 14
MAX_TOOL_CALLS = 48
MAX_CONTEXT_CHARS = 300_000     # tool output sent to the model; beyond this the agent must submit
MAX_OUTPUT_TOKENS = 6000
MAX_OUTPUT_TOKENS_REASONING = 24000
MIN_INVESTIGATION_CALLS_BASE = 4   # beyond the pre-run overview / findings / physics
MIN_INVESTIGATION_CALLS_CAP = 16
LABEL = "Agentic Analysis: an AI agent investigated this flight through code tools. Every number was computed by code."

CONSENT_TEXT = (
    "Agentic Analysis sends the results of code tools to OpenAI ({model}): the flight overview, detector findings, physics "
    "checks and the statistics the agent asks for (values, times, units). The raw log, GPS coordinates and engineer labels "
    "are never sent. Without consent you still get every code finding, but no investigation or written report."
)


def consent_text() -> str:
    return CONSENT_TEXT.format(model=get_model_name())


def _classify(exc: BaseException) -> str:
    blob = f"{type(exc).__name__} {exc!r}".lower()
    if "auth" in blob or "permission" in blob or "invalid_api_key" in blob or isinstance(exc, LLMError):
        return "auth"
    if "ratelimit" in blob or "rate limit" in blob or "429" in blob:
        return "rate_limit"
    if "timeout" in blob:
        return "timeout"
    if "context_length" in blob or "maximum context" in blob:
        return "context"
    if any(k in blob for k in ("connection", "internalserver", "serviceunavailable", "502", "503", "504")):
        return "transient"
    return "other"


API_MESSAGE = {
    "auth": "The AI service rejected the server's credentials.",
    "rate_limit": "The AI service is rate-limiting requests. Try again in a minute.",
    "timeout": "The AI service did not answer in time.",
    "context": "The investigation grew larger than the model's context window.",
    "transient": "The AI service could not be reached.",
    "other": "The AI request failed ({type}).",
}


def _assistant_message(msg: Any) -> Dict[str, Any]:
    calls = [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"}}
             for tc in (msg.tool_calls or [])]
    if calls:
        return {"role": "assistant", "content": msg.content or None, "tool_calls": calls}
    return {"role": "assistant", "content": msg.content or "(no reply)"}  # the API rejects an empty assistant message


def _params(model: str) -> Dict[str, Any]:
    if model.startswith(("gpt-4", "gpt-3")):
        return {"max_completion_tokens": MAX_OUTPUT_TOKENS, "temperature": 0.1}
    # reasoning models: no temperature, and the token budget also covers their reasoning
    return {"max_completion_tokens": MAX_OUTPUT_TOKENS_REASONING}


VERIFY_MARGIN_S = 30.0
WINDOW_ARGS = (("t_start", "t_end"), ("a_start", "a_end"), ("b_start", "b_end"))


def _span(f: Dict[str, Any]) -> Optional[tuple]:
    s, e = f.get("t_start"), f.get("t_end")
    try:
        return float(s), float(e)
    except (TypeError, ValueError):
        return None


def unverified(findings: List[Dict[str, Any]], calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Findings no investigation call has looked at yet: neither their time window (+-30 s) nor their evidence signals."""
    out = []
    for f in findings:
        topics = {e.get("topic") for e in f.get("evidence") or []}
        span = _span(f)
        whole = f.get("window") == "whole_flight"
        hit = False
        for c in calls:
            if not c.get("ok"):
                continue
            a = c.get("args") or {}
            if a.get("topic") in topics and whole:
                hit = True
            elif f.get("code") == "MOTOR_ESC" and c["tool"] == "motor_outputs":
                hit = True
            elif span is not None:
                for lo_k, hi_k in WINDOW_ARGS:
                    lo, hi = a.get(lo_k), a.get(hi_k)
                    if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and \
                            lo <= span[1] + VERIFY_MARGIN_S and hi >= span[0] - VERIFY_MARGIN_S:
                        hit = True
            if hit:
                break
        if not hit:
            out.append(f)
    return out


def _nudge(pending: List[Dict[str, Any]], made: int, need: int) -> str:
    parts = []
    if made < need:
        parts.append(f"You have made {made} investigation calls; make at least {need} before the report can be submitted.")
    if pending:
        parts.append("Not yet verified on the raw signals: " + "; ".join(
            f"{f['id']} ({f.get('code')}, {f.get('when') or ''} {f.get('title') or ''})".strip()
            for f in pending[:10]) + ". Examine each over its own window or evidence signals.")
    parts.append("Also look beyond the findings (protocol step 3).")
    return " ".join(parts)


def _pilot_window(pilot: Optional[Dict[str, Any]], analysis: Dict[str, Any], source: Source) -> Optional[Dict[str, Any]]:
    if not pilot:
        return None
    from src.agent.symptoms import search_window
    tl = analysis.get("timeline") or {}
    return search_window(pilot, tl.get("takeoff_t"), tl.get("landing_t"), source.bounds()[1])


def investigate(source: Source, analysis: Dict[str, Any], findings: List[Dict[str, Any]], *,
                pilot_report: Optional[Dict[str, Any]] = None, client: Any = None, input_fingerprint: str = "") -> Dict[str, Any]:
    model = get_model_name()
    kind = source.kind
    pilot = None
    if pilot_report and pilot_report.get("symptom"):
        pilot = {**pilot_report, "symptom_label": SYMPTOMS.get(pilot_report["symptom"], SYMPTOMS["other"])["label"]}
    prov: Dict[str, Any] = {"model": model, "prompt_version": prompt.PROMPT_VERSION, "kind": kind,
                            "input_fingerprint": input_fingerprint, "pilot_report": pilot,
                            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    base = {"label": LABEL, "provenance": prov, "report": None, "trace": [], "message": "", "failure": None}
    if client is None and not api_configured():
        return {**base, "status": "unavailable", "message": "Agentic Analysis is not configured on the server (no OpenAI key). "
                                                             "The code findings below are complete without it."}
    try:
        c = client or OpenAIChatClient()
    except LLMError:
        return {**base, "status": "unavailable", "message": "Agentic Analysis is not configured on the server."}

    tb = Toolbox(source, analysis, pilot)
    specs = tb.specs()
    messages: List[Dict[str, Any]] = [{"role": "system", "content": prompt.SYSTEM},
                                      {"role": "user", "content": prompt.kickoff(pilot, _pilot_window(pilot, analysis, source))}]
    pre = [("get_flight_overview", {}), ("get_findings", {}), ("get_physics", {})]
    if pilot:
        pre.append(("match_pilot_report", {}))
    pre_calls, pre_msgs = [], []
    for i, (name, args) in enumerate(pre):
        ref, out = tb.call(name, args)
        cid = f"call_pre_{i}"
        pre_calls.append({"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}})
        pre_msgs.append({"role": "tool", "tool_call_id": cid, "content": json.dumps(out, default=str, separators=(",", ":"))})
    messages.append({"role": "assistant", "content": None, "tool_calls": pre_calls})
    messages += pre_msgs
    sent_chars = sum(len(m["content"]) for m in pre_msgs)

    raw_report: Optional[Dict[str, Any]] = None
    tokens = {"prompt": 0, "completion": 0}
    t0 = time.perf_counter()
    rounds = 0
    nudged = 0
    n_pre = len(tb.calls)
    need = min(MIN_INVESTIGATION_CALLS_CAP, MIN_INVESTIGATION_CALLS_BASE + 2 * len(findings))
    explore_specs = [s for s in specs if s["function"]["name"] != "submit_report"]
    for rounds in range(1, MAX_ROUNDS + 1):
        force = rounds == MAX_ROUNDS or len(tb.calls) >= MAX_TOOL_CALLS or sent_chars >= MAX_CONTEXT_CHARS
        made = sum(1 for c in tb.calls[n_pre:] if c["ok"])
        pending = unverified(findings, tb.calls[n_pre:])
        ready = made >= need and not pending
        # submit_report is withheld until the agent has investigated: every finding examined, minimum calls made.
        late = rounds >= MAX_ROUNDS - 2
        kwargs: Dict[str, Any] = {"model": model, "messages": messages, **_params(model)}
        if force:
            kwargs.update(tools=specs, tool_choice={"type": "function", "function": {"name": "submit_report"}},
                          parallel_tool_calls=False)
        elif ready or late:
            kwargs.update(tools=specs, tool_choice="auto")
        else:
            kwargs.update(tools=explore_specs, tool_choice="required")
        try:
            resp = c.create(**kwargs)
        except Exception as exc:  # network / SDK errors must not lose the code findings
            kind_err = _classify(exc)
            if kind_err in ("rate_limit", "transient", "timeout") and rounds < MAX_ROUNDS:
                time.sleep(1.5 * rounds)
                try:
                    resp = c.create(**kwargs)
                except Exception as exc2:
                    exc, kind_err = exc2, _classify(exc2)
                    resp = None
            else:
                resp = None
            if resp is None:
                detail = str(getattr(exc, "message", "") or exc)[:300]  # API error text; never contains the key
                log.warning("agent api error round=%d kind=%s type=%s detail=%s", rounds, kind_err, type(exc).__name__, detail)
                prov.update(rounds=rounds, latency_ms=round((time.perf_counter() - t0) * 1000), tokens=tokens)
                return {**base, "status": "error", "trace": tb.calls,
                        "message": API_MESSAGE[kind_err].format(type=type(exc).__name__) + " The code findings are unaffected.",
                        "failure": {"reason": f"api_{kind_err}", "round": rounds, "detail": detail}}
        usage = getattr(resp, "usage", None)
        tokens["prompt"] += int(getattr(usage, "prompt_tokens", 0) or 0)
        tokens["completion"] += int(getattr(usage, "completion_tokens", 0) or 0)
        msg = resp.choices[0].message
        messages.append(_assistant_message(msg))
        if not msg.tool_calls:
            nudged += 1
            if nudged > 2:
                break
            messages.append({"role": "user", "content": "Continue the investigation with tools, or call submit_report if it is complete."})
            continue
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
                if not isinstance(args, dict):
                    raise ValueError
            except (json.JSONDecodeError, ValueError):
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": '{"error":"arguments were not a JSON object"}'})
                continue
            if name == "submit_report":
                raw_report = args
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": '{"ok":true}'})
                continue
            _, out = tb.call(name, args)
            content = json.dumps(out, default=str, separators=(",", ":"))
            sent_chars += len(content)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
        if raw_report is not None:
            break
        made = sum(1 for c in tb.calls[n_pre:] if c["ok"])
        pending = unverified(findings, tb.calls[n_pre:])
        if made < need or pending:
            messages.append({"role": "user", "content": _nudge(pending, made, need)})

    prov.update(rounds=rounds, tool_calls=len(tb.calls), latency_ms=round((time.perf_counter() - t0) * 1000), tokens=tokens,
                tool_output_chars=sent_chars, finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    if raw_report is None:
        return {**base, "status": "incomplete", "trace": tb.calls,
                "message": "The agent did not submit a report within its limits. The code findings are unaffected.",
                "failure": {"reason": "no_report", "rounds": rounds}}
    ids_source = findings
    report = validate(raw_report, findings=ids_source, tool_results=tb.results, analysis=analysis)
    partial = bool(report["rejected"]) or bool(report["unreviewed_findings"]) or not report["headline"]
    log.info("agent ok rounds=%d calls=%d rejected=%d unreviewed=%d tokens=%s", rounds, len(tb.calls),
             len(report["rejected"]), len(report["unreviewed_findings"]), tokens)
    return {**base, "status": "partial" if partial else "ok", "report": report, "trace": tb.calls,
            "message": ("Some statements were removed because they could not be traced to tool results, or some findings "
                        "were not reviewed by the agent." if partial else "")}
