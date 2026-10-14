"""Reports — validated JSON and CSV exports from the latest analysis."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.report_export import (
    events_to_csv_bytes,
    report_to_json_bytes,
    windows_to_csv_bytes,
)
from src.report_schema import SCHEMA_VERSION, validate_report
from src.session_state import (
    FLIGHT_ID_KEY,
    get_analysis,
    get_analysis_windows,
    get_events_df,
    get_report,
    has_flight,
)

st.set_page_config(page_title="Reports · UAV Sentinel", layout="wide")

st.title("Reports")
st.caption(f"Schema {SCHEMA_VERSION} · detector + optional ai_analysis")

if not has_flight(st.session_state):
    st.info("Upload a flight and run analysis first.")
    st.page_link("app.py", label="Go to upload", icon="⬆")
    st.stop()

analysis = get_analysis(st.session_state)
report = get_report(st.session_state)
windows = get_analysis_windows(st.session_state)
events_df = get_events_df(st.session_state)
flight_id = st.session_state.get(FLIGHT_ID_KEY, "flight")

if analysis is None or report is None:
    st.warning("No analysis report in session. Run **Flight Analysis** first.")
    st.page_link("pages/1_Flight_Analysis.py", label="Open Flight Analysis", icon="📊")
    st.stop()

st.subheader(f"Flight `{flight_id}`")
st.write(
    f"Status: `{report.analysis_status}` · "
    f"windows={report.summary.total_windows} · "
    f"flagged={report.summary.flagged_windows} · "
    f"events={report.summary.event_count}"
)

try:
    validate_report(report)
    st.success("JSON report validates against schema 1.0")
except Exception as exc:
    st.error(f"Report failed schema validation: {exc}")
    st.stop()

with st.expander("Grouping policy", expanded=False):
    st.json(report.grouping_policy)

with st.expander("Report preview (truncated)", expanded=True):
    preview = report.to_json_dict()
    # Keep preview readable
    if len(preview.get("events", [])) > 3:
        preview = dict(preview)
        preview["events"] = preview["events"][:3]
        preview["_preview_note"] = "Showing first 3 events only in UI preview"
    st.code(json.dumps(preview, indent=2)[:8000], language="json")

st.subheader("Downloads")
c1, c2, c3 = st.columns(3)
with c1:
    st.download_button(
        "Download JSON report",
        data=report_to_json_bytes(report),
        file_name=f"{flight_id}_report.json",
        mime="application/json",
        width='stretch',
    )
with c2:
    st.download_button(
        "Download windows CSV",
        data=windows_to_csv_bytes(windows),
        file_name=f"{flight_id}_windows.csv",
        mime="text/csv",
        width='stretch',
    )
with c3:
    st.download_button(
        "Download events CSV",
        data=events_to_csv_bytes(events_df),
        file_name=f"{flight_id}_events.csv",
        mime="text/csv",
        width='stretch',
    )

if report.ai_analysis is None:
    st.info(
        "`ai_analysis` is null until you generate a diagnostic on the "
        "AI Diagnostics page."
    )
    st.page_link("pages/2_AI_Diagnostics.py", label="Open AI Diagnostics", icon="💬")
else:
    st.success("`ai_analysis` is attached to this report export.")
