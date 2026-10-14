"""Flight-isolated chat history helpers (no Streamlit dependency)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.diagnostic_schema import ChatAnswer, DiagnosticReport
from src.llm_service import LLMService
from src.report_schema import FlightReport

MAX_HISTORY_TURNS = 16  # user+assistant messages


def empty_chat_state(flight_id: str) -> Dict[str, Any]:
    return {
        "flight_id": flight_id,
        "messages": [],  # {role, content} for UI; assistant may store structured
        "selected_event_id": None,
    }


def ensure_chat_for_flight(session: Dict[str, Any], flight_id: str, key: str) -> Dict[str, Any]:
    """Return chat state for flight_id, resetting if the flight changed."""
    state = session.get(key)
    if not isinstance(state, dict) or state.get("flight_id") != flight_id:
        state = empty_chat_state(flight_id)
        session[key] = state
    return state


def reset_chat(session: Dict[str, Any], flight_id: str, key: str) -> Dict[str, Any]:
    state = empty_chat_state(flight_id)
    session[key] = state
    return state


def append_turn(state: Dict[str, Any], role: str, content: str) -> None:
    msgs: List[Dict[str, str]] = list(state.get("messages") or [])
    msgs.append({"role": role, "content": content})
    if len(msgs) > MAX_HISTORY_TURNS:
        msgs = msgs[-MAX_HISTORY_TURNS:]
    state["messages"] = msgs


def history_for_llm(state: Dict[str, Any]) -> List[Dict[str, str]]:
    """Pass only role/content pairs; strip any accidental structured blobs."""
    out = []
    for m in state.get("messages") or []:
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        out.append({"role": role, "content": str(m.get("content", ""))})
    return out


def ask_followup(
    service: LLMService,
    *,
    report: FlightReport,
    chat_state: Dict[str, Any],
    user_message: str,
    diagnostic: Optional[DiagnosticReport] = None,
    windows=None,
    aligned=None,
) -> ChatAnswer:
    if chat_state.get("flight_id") != report.flight_id:
        raise ValueError(
            "Chat state flight_id does not match the active report — "
            "refusing to mix flight contexts."
        )
    selected = chat_state.get("selected_event_id")
    selected_ids = [selected] if selected else None
    answer = service.chat(
        report=report,
        user_message=user_message,
        history=history_for_llm(chat_state),
        diagnostic=diagnostic,
        windows=windows,
        aligned=aligned,
        selected_event_ids=selected_ids,
    )
    if answer.flight_id != report.flight_id:
        # Force isolation even if the model mis-labels
        answer = answer.model_copy(update={"flight_id": report.flight_id})
    append_turn(chat_state, "user", user_message.strip())
    append_turn(chat_state, "assistant", answer.answer)
    return answer
