"""Flight Analysis — scores, events, telemetry evidence, exports."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.artifact_loader import ArtifactError, load_artifacts, missing_artifacts
from src.event_builder import DEFAULT_MAX_GAP_S
from src.inference_service import analyze_flight
from src.report_export import (
    events_to_csv_bytes,
    report_to_json_bytes,
    windows_to_csv_bytes,
)
from src.session_state import (
    FLIGHT_ID_KEY,
    FLIGHT_KEY,
    UPLOAD_META_KEY,
    get_aligned,
    get_analysis,
    get_analysis_windows,
    get_events,
    get_events_df,
    get_report,
    has_flight,
    store_analysis,
)
from src.visualization import (
    baro_context_figure,
    per_feature_error_figure,
    score_timeline_figure,
    telemetry_around_event_figure,
)

st.set_page_config(page_title="Flight Analysis · UAV Sentinel", layout="wide")

st.title("Flight Analysis")
st.caption("Phase 3 · Events, dashboard & reports")

# —— Artifacts ——————————————————————————————————————————————————————————————
missing = missing_artifacts()
if missing:
    st.error(
        "Model artifacts are incomplete. Missing: **"
        + ", ".join(missing)
        + "**. See `uav_model/ARTIFACTS.md`."
    )
    st.stop()

try:
    arts = load_artifacts()
    st.success(
        f"Artifacts · **{arts.variant}** · {arts.n_features} features · "
        f"threshold **{arts.threshold:.5f}** · "
        f"window {arts.win_samples}/{arts.stride_samples}"
    )
except ArtifactError as exc:
    st.error(str(exc))
    st.stop()

st.info(
    "LSTM scores **12 conservative IMU/ATT features** only. BARO is shown for "
    "context and quality warnings — not as LSTM anomaly evidence. "
    "Zero flagged windows is not a definitive healthy-flight verdict."
)

if not has_flight(st.session_state):
    st.warning("No flight in session. Upload telemetry on the home page first.")
    st.page_link("app.py", label="Go to upload", icon="⬆")
    st.stop()

flight_id = st.session_state[FLIGHT_ID_KEY]
frames = st.session_state[FLIGHT_KEY]
meta = st.session_state.get(UPLOAD_META_KEY, {})

st.subheader(f"Flight `{flight_id}`")
st.caption(
    f"Source: {meta.get('source', '—')} · "
    f"Files: {', '.join(f'{k}={v}' for k, v in (meta.get('filenames') or {}).items())}"
)

with st.expander("Event grouping settings", expanded=False):
    max_gap_s = st.number_input(
        "Max gap between flagged windows (s)",
        min_value=0.0,
        max_value=30.0,
        value=float(DEFAULT_MAX_GAP_S),
        step=0.5,
        help=(
            "Within one session, flagged windows whose intervals overlap or "
            "are separated by at most this gap merge into one event. "
            "Events never cross session boundaries."
        ),
    )
    st.caption(
        "Event duration = max(t_end) − min(t_start) of member windows "
        "(not n_windows × window length)."
    )

if st.button("Run Anomaly Analysis", type="primary", width='stretch'):
    with st.spinner("Aligning, scoring, grouping events, building report…"):
        result = analyze_flight(
            frames,
            flight_id=flight_id,
            upload_warnings=list(meta.get("warnings") or []),
            max_gap_s=float(max_gap_s),
        )
        store_analysis(st.session_state, result)

analysis = get_analysis(st.session_state)
windows = get_analysis_windows(st.session_state)
events = get_events(st.session_state)
events_df = get_events_df(st.session_state)
aligned = get_aligned(st.session_state)
report = get_report(st.session_state)

if analysis is None:
    st.markdown("Click **Run Anomaly Analysis** to score this flight.")
    st.stop()

for err in analysis.get("errors") or []:
    st.error(err)
for warn in analysis.get("warnings") or []:
    st.warning(warn)

if analysis.get("status") == "insufficient_data":
    st.stop()
if not analysis.get("ok"):
    st.stop()

# —— Summary ————————————————————————————————————————————————————————————————
rate = analysis.get("flagged_rate")
rate_label = f"{rate * 100:.2f}%" if rate is not None else "—"
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Windows scored", f"{analysis['n_windows']:,}")
c2.metric("Flagged windows", f"{analysis['n_flagged']:,}")
c3.metric("Flagged rate", rate_label)
c4.metric("Anomaly events", f"{analysis.get('n_events', 0):,}")
c5.metric("Threshold", f"{analysis['threshold']:.5f}")

# —— Timeline ————————————————————————————————————————————————————————————————
st.subheader("Anomaly score timeline")
event_ids = [e.event_id for e in events]
selected = st.selectbox(
    "Highlight event",
    options=["(none)"] + event_ids,
    index=0,
)
selected_id = None if selected == "(none)" else selected
fig = score_timeline_figure(
    windows,
    float(analysis["threshold"]),
    events=events,
    selected_event_id=selected_id,
)
st.plotly_chart(fig, width='stretch')

if analysis.get("n_flagged", 0) == 0:
    st.caption(
        "No windows crossed the threshold — status `no_anomaly`. "
        "This is not a certified healthy-flight label."
    )

# —— Event explorer ——————————————————————————————————————————————————————————
st.subheader("Event explorer")
if events_df is not None and len(events_df):
    st.dataframe(events_df, width='stretch', height=240)

    detail_id = st.selectbox(
        "Inspect event",
        options=event_ids,
        index=event_ids.index(selected_id) if selected_id in event_ids else 0,
    )
    ev = next(e for e in events if e.event_id == detail_id)

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Start (s)", f"{ev.start_time_s:.2f}")
    d2.metric("End (s)", f"{ev.end_time_s:.2f}")
    d3.metric("Duration (s)", f"{ev.duration_s:.2f}")
    d4.metric("Peak score", f"{ev.peak_score:.4f}")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(per_feature_error_figure(ev), width='stretch')
    with right:
        st.plotly_chart(
            telemetry_around_event_figure(
                aligned, ev, list(analysis.get("features") or [])
            ),
            width='stretch',
        )
    st.plotly_chart(baro_context_figure(aligned, ev), width='stretch')
    st.caption(
        "Affected signals are ranked by peak per-feature reconstruction error "
        "when available — not a proven root cause."
    )
else:
    st.write("No anomaly events for this flight.")

# —— Window table ————————————————————————————————————————————————————————————
st.subheader("Window-level results")
if windows is not None and len(windows):
    show_cols = [
        c
        for c in (
            "window_index",
            "session_id",
            "t_start",
            "t_end",
            "anomaly_score",
            "threshold",
            "is_anomaly",
        )
        if c in windows.columns
    ]
    st.dataframe(windows[show_cols], width='stretch', height=300)

# —— Downloads ——————————————————————————————————————————————————————————————
st.subheader("Downloads")
col_a, col_b, col_c = st.columns(3)
with col_a:
    if report is not None:
        st.download_button(
            "JSON report",
            data=report_to_json_bytes(report),
            file_name=f"{flight_id}_report.json",
            mime="application/json",
            width='stretch',
        )
with col_b:
    st.download_button(
        "Windows CSV",
        data=windows_to_csv_bytes(windows),
        file_name=f"{flight_id}_windows.csv",
        mime="text/csv",
        width='stretch',
    )
with col_c:
    st.download_button(
        "Events CSV",
        data=events_to_csv_bytes(events_df),
        file_name=f"{flight_id}_events.csv",
        mime="text/csv",
        width='stretch',
    )

st.divider()
st.page_link(
    "pages/2_AI_Diagnostics.py",
    label="Request AI analysis (GPT-4o mini)",
    icon="💬",
)
st.caption(
    "AI interprets this report’s evidence only — it does not change scores or flags. "
    "Requires OPENAI_API_KEY in `.env`."
)
