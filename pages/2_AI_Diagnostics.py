"""AI Diagnostics — GPT-4o mini interpretation & flight-specific chat."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_CONFIDENCE_PLAIN = {
    "high": "stronger possibility",
    "medium": "possible",
    "low": "weaker possibility",
}


def _confidence_label(value: str) -> str:
    return _CONFIDENCE_PLAIN.get(value, value)

from src.chat_service import ask_followup
from src.diagnostic_trace import build_diagnostic_trace, format_diagnostic_trace
from src.evidence_builder import build_evidence_pack
from src.llm_service import (
    LLMError,
    LLMService,
    api_configured,
    attach_ai_to_report,
    get_model_name,
)
from src.report_export import report_to_json_bytes
from src.session_state import (
    FLIGHT_ID_KEY,
    clear_chat,
    get_ai_diagnostic,
    get_ai_error,
    get_aligned,
    get_analysis_windows,
    get_events,
    get_or_create_chat,
    get_report,
    has_flight,
    store_ai_diagnostic,
    store_ai_error,
)

st.set_page_config(page_title="AI Diagnostics · UAV Sentinel", layout="wide")

st.title("AI Diagnostics")
st.caption("Plain-language explanations — detector scores and flags are never changed")

st.info(
    "The automatic check is the only source of scores and flags; the assistant "
    "just explains them in everyday words. Its possible causes are unverified — "
    "they are not confirmed faults."
)

if not has_flight(st.session_state):
    st.warning("No flight in session. Upload and analyze a flight first.")
    st.page_link("app.py", label="Go to upload", icon="⬆")
    st.stop()

report = get_report(st.session_state)
flight_id = st.session_state[FLIGHT_ID_KEY]
windows = get_analysis_windows(st.session_state)
aligned = get_aligned(st.session_state)
events = get_events(st.session_state)

if report is None:
    st.warning("No Phase 3 report yet. Run anomaly analysis first.")
    st.page_link("pages/1_Flight_Analysis.py", label="Open Flight Analysis", icon="📊")
    st.stop()

st.subheader(f"Flight `{flight_id}`")
st.caption(
    f"Status `{report.analysis_status}` · "
    f"{report.summary.total_windows} windows · "
    f"{report.summary.flagged_windows} flagged · "
    f"{report.summary.event_count} events · "
    f"model `{get_model_name()}`"
)

# Compact pipeline trace (no API keys / raw telemetry)
_preview_pack = build_evidence_pack(
    report, windows=windows, aligned=aligned
)
_trace = build_diagnostic_trace(
    report, windows=windows, evidence_pack=_preview_pack
)
with st.expander("Evidence used by the assistant (technical view)", expanded=report.summary.event_count == 0):
    st.caption("The check's numbers behind the explanation — kept for transparency.")
    st.code(format_diagnostic_trace(_trace), language="text")
    if report.summary.flagged_windows == 0:
        st.caption(
            "No 3-second moment of this flight crossed the alert level — a valid "
            "'nothing unusual' result. The assistant receives the score story and "
            "the highest scores (all still below the alert level); it never "
            "fabricates events."
        )

if not api_configured():
    st.warning(
        "OPENAI_API_KEY is not configured. Detector results and JSON exports "
        "still work. Copy `.env.example` → `.env` and set your key to enable AI."
    )

event_ids = [e.event_id for e in events]
chat_state = get_or_create_chat(st.session_state, flight_id)
selected = st.selectbox(
    "Focus event (optional)",
    options=["(all events)"] + event_ids,
    index=0,
)
chat_state["selected_event_id"] = None if selected == "(all events)" else selected

col_run, col_reset = st.columns(2)
with col_run:
    run_ai = st.button(
        "Generate AI diagnostic report",
        type="primary",
        width='stretch',
        disabled=not api_configured(),
    )
with col_reset:
    if st.button("New chat / clear AI session", width='stretch'):
        clear_chat(st.session_state, flight_id)
        st.session_state.pop("ai_diagnostic", None)
        st.session_state.pop("ai_error", None)
        st.rerun()

if run_ai:
    with st.spinner("Calling GPT-4o mini…"):
        try:
            service = LLMService()
            selected_ids = (
                [chat_state["selected_event_id"]]
                if chat_state.get("selected_event_id")
                else None
            )
            diagnostic = service.generate_diagnostic(
                report,
                windows=windows,
                aligned=aligned,
                selected_event_ids=selected_ids,
            )
            report_with_ai = attach_ai_to_report(report, diagnostic)
            store_ai_diagnostic(st.session_state, diagnostic, report_with_ai)
            pack = service.last_evidence_pack or _preview_pack
            sent_trace = build_diagnostic_trace(
                report, windows=windows, evidence_pack=pack
            )
            st.caption(
                "Evidence sent · "
                + format_diagnostic_trace(sent_trace).replace("\n", " · ")
            )
            usage = service.last_usage
            if usage:
                st.caption(
                    f"Latency {usage.latency_ms:.0f} ms · "
                    f"tokens prompt={usage.prompt_tokens} "
                    f"completion={usage.completion_tokens}"
                )
        except LLMError as exc:
            store_ai_error(st.session_state, str(exc))
            st.error(str(exc))
        except Exception as exc:
            store_ai_error(st.session_state, "Unexpected AI error. Detector results are intact.")
            st.error("Unexpected AI error. Detector results are intact.")
            st.caption(str(type(exc).__name__))

err = get_ai_error(st.session_state)
if err and not run_ai:
    st.error(err)

diagnostic = get_ai_diagnostic(st.session_state)

# —— Diagnostic panel ————————————————————————————————————————————————————————
st.subheader("AI diagnostic report")
if diagnostic is None:
    st.write("No AI report yet. Generate one above when an API key is configured.")
else:
    if diagnostic.insufficient_evidence:
        st.warning("The assistant doesn't have enough evidence to point to a specific problem yet.")
    st.markdown(f"**Summary:** {diagnostic.summary}")
    if diagnostic.overall_limitations:
        with st.expander("Limitations", expanded=False):
            for item in diagnostic.overall_limitations:
                st.write(f"- {item}")
    for finding in diagnostic.findings:
        with st.expander(f"Finding · {finding.event_id}", expanded=True):
            st.write("**What the system observed**")
            for f in finding.observed_facts or ["(none)"]:
                st.write(f"- {f}")
            st.write("**What it might mean**")
            if not finding.possible_causes:
                st.write("- (none)")
            for cause in finding.possible_causes:
                st.markdown(
                    f"- **{cause.hypothesis}** · {_confidence_label(cause.confidence)}"
                )
                if cause.supporting_evidence:
                    st.caption("Signs for: " + "; ".join(cause.supporting_evidence))
                if cause.contradicting_evidence:
                    st.caption(
                        "Signs against: " + "; ".join(cause.contradicting_evidence)
                    )
            st.write("**What it can't tell you**")
            for u in finding.unknowns or ["(none)"]:
                st.write(f"- {u}")
            st.write("**What you can check**")
            for c in finding.recommended_checks or ["(none)"]:
                st.write(f"- {c}")

    st.download_button(
        "Download AI diagnostic JSON",
        data=json.dumps(diagnostic.to_json_dict(), indent=2).encode("utf-8"),
        file_name=f"{flight_id}_ai_diagnostic.json",
        mime="application/json",
        width='stretch',
    )
    # Combined report (detector + ai_analysis)
    current_report = get_report(st.session_state)
    if current_report is not None and current_report.ai_analysis is not None:
        st.download_button(
            "Download full flight report (with ai_analysis)",
            data=report_to_json_bytes(current_report),
            file_name=f"{flight_id}_report_with_ai.json",
            mime="application/json",
            width='stretch',
            key="dl_full_with_ai",
        )

# —— Chat ————————————————————————————————————————————————————————————————————
st.subheader("Flight chat")
st.caption(
    "This chat belongs to this flight. Switching or resetting flights clears it. "
    "Earlier possibilities the assistant mentioned are not treated as facts."
)

for msg in chat_state.get("messages") or []:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input(
    "Ask a follow-up about this flight…",
    disabled=not api_configured(),
)
if prompt:
    if diagnostic is None:
        st.warning("Generate an AI diagnostic report first for grounded chat.")
    else:
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    service = LLMService()
                    answer = ask_followup(
                        service,
                        report=get_report(st.session_state) or report,
                        chat_state=chat_state,
                        user_message=prompt,
                        diagnostic=diagnostic,
                        windows=windows,
                        aligned=aligned,
                    )
                    st.write(answer.answer)
                    if answer.caveats:
                        st.caption("Caveats: " + "; ".join(answer.caveats))
                    if answer.insufficient_evidence:
                        st.caption("Not enough evidence for a specific answer to this one.")
                except LLMError as exc:
                    st.error(str(exc))
                except ValueError as exc:
                    st.error(str(exc))

st.divider()
st.page_link("pages/1_Flight_Analysis.py", label="Back to Flight Analysis", icon="📊")
st.page_link("pages/3_Reports.py", label="Reports & exports", icon="📄")
