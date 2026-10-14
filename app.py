"""UAV Sentinel — Telemetry upload (Phase 1) + analysis entry point."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from src.session_state import (
    get_flight_summary,
    has_flight,
    reset_flight,
    store_upload,
)
from src.upload import load_individual_csvs, load_zip_archive

st.set_page_config(
    page_title="UAV Sentinel",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# —— Styles ————————————————————————————————————————————————————————————————
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg: #0c1117;
  --surface: #141b24;
  --border: #243040;
  --text: #e8eef6;
  --muted: #8b9bb0;
  --accent: #3d9a7a;
  --accent-dim: #2a6b55;
  --warn: #c9a227;
  --error: #c45c5c;
  --imu: #5b8def;
  --att: #c97b4a;
  --baro: #6bbf8a;
}

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
}

.block-container { padding-top: 1.5rem; max-width: 1100px; }

.brand-block {
  margin-bottom: 1.75rem;
  padding-bottom: 1.25rem;
  border-bottom: 1px solid var(--border);
}
.brand-name {
  font-size: 1.85rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--text);
  margin: 0;
}
.brand-tag {
  font-size: 0.95rem;
  color: var(--muted);
  margin: 0.15rem 0 0 0;
}
.phase-pill {
  display: inline-block;
  margin-top: 0.65rem;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem;
  font-weight: 500;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--accent);
  border: 1px solid var(--accent-dim);
  padding: 0.2rem 0.55rem;
  border-radius: 4px;
}

.stream-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1rem 1.1rem;
  margin-bottom: 0.75rem;
}
.stream-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.75rem;
  font-weight: 500;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 0.35rem;
}
.stream-imu .stream-label { color: var(--imu); }
.stream-att .stream-label { color: var(--att); }
.stream-baro .stream-label { color: var(--baro); }
.stream-desc { color: var(--muted); font-size: 0.88rem; margin: 0; }

.status-ok {
  color: var(--accent);
  font-weight: 600;
}
.status-err {
  color: var(--error);
  font-weight: 600;
}
.meta-mono {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.8rem;
  color: var(--muted);
}
</style>
""",
    unsafe_allow_html=True,
)


def _fmt_bytes(n: int | None) -> str:
    if n is None:
        return "—"
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.2f} MB"


def _fmt_hz(hz: float | None) -> str:
    if hz is None:
        return "—"
    return f"{hz:.2f} Hz"


def render_header() -> None:
    st.markdown(
        """
<div class="brand-block">
  <p class="brand-name">UAV Sentinel</p>
  <p class="brand-tag">Telemetry Intelligence</p>
  <span class="phase-pill">Phase 4 · Detect · Diagnose · Chat</span>
</div>
""",
        unsafe_allow_html=True,
    )


def render_stream_overview() -> None:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
<div class="stream-card stream-imu">
  <div class="stream-label">IMU</div>
  <p class="stream-desc">Gyroscope and acceleration</p>
</div>
""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
<div class="stream-card stream-att">
  <div class="stream-label">ATT</div>
  <p class="stream-desc">Orientation and targets</p>
</div>
""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
<div class="stream-card stream-baro">
  <div class="stream-label">BARO</div>
  <p class="stream-desc">Pressure, altitude, temperature</p>
</div>
""",
            unsafe_allow_html=True,
        )


def render_validation_summary(validation: dict) -> None:
    streams = validation.get("streams", {})
    for stream in ("IMU", "ATT", "BARO"):
        sv = streams.get(stream)
        if not sv:
            continue
        with st.expander(
            f"{stream} — {sv.get('n_rows', 0):,} rows · "
            f"{'OK' if sv.get('ok') else 'FAILED'}",
            expanded=True,
        ):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Rows", f"{sv.get('n_rows', 0):,}")
            m2.metric("Size", _fmt_bytes(sv.get("file_size_bytes")))
            m3.metric("Est. rate", _fmt_hz(sv.get("est_hz")))
            m4.metric(
                "Duration",
                f"{sv['duration_s']:.1f} s" if sv.get("duration_s") is not None else "—",
            )

            st.caption(
                f"File: `{sv.get('filename') or '—'}` · "
                f"Timestamp: `{sv.get('timestamp_column') or '—'}` · "
                f"Missing frac: {sv.get('missing_value_frac', 0):.2%} · "
                f"Dup TS: {sv.get('duplicate_timestamps', 0)} · "
                f"Non-mono: {sv.get('non_monotonic_timestamps', 0)}"
            )

            if sv.get("t_start") is not None and sv.get("t_end") is not None:
                st.caption(
                    f"Time range (s): {sv['t_start']:.3f} → {sv['t_end']:.3f}"
                )

            cols = sv.get("columns") or []
            st.write("**Detected columns**")
            st.code(", ".join(cols) if cols else "(none)", language=None)

            if sv.get("warnings"):
                for w in sv["warnings"]:
                    st.warning(w)
            if sv.get("errors"):
                for e in sv["errors"]:
                    st.error(e)

            sample = sv.get("sample_rows") or []
            if sample:
                st.write("**Sample (first 5 rows)**")
                st.dataframe(pd.DataFrame(sample), width='stretch')


def main() -> None:
    if "flight_status" not in st.session_state:
        reset_flight(st.session_state)

    render_header()

    with st.sidebar:
        st.markdown("**Navigation**")
        st.page_link("app.py", label="New flight upload", icon="⬆")
        st.page_link("pages/1_Flight_Analysis.py", label="Flight Analysis", icon="📊")
        st.page_link("pages/2_AI_Diagnostics.py", label="AI Diagnostics", icon="💬")
        st.page_link("pages/3_Reports.py", label="Reports", icon="📄")
        st.divider()
        if has_flight(st.session_state):
            summary = get_flight_summary(st.session_state)
            st.success(f"Flight loaded\n\n`{summary['flight_id']}`")
            if st.button("Reset flight", width='stretch'):
                reset_flight(st.session_state)
                st.rerun()
        else:
            st.caption("No flight in session.")

    st.subheader("New flight analysis")
    st.write(
        "Upload the telemetry streams from a single flight to begin. "
        "After validation, open **Flight Analysis** and click "
        "**Run Anomaly Analysis** to score the flight with the conservative LSTM."
    )

    render_stream_overview()

    flight_id_input = st.text_input(
        "Flight ID (optional)",
        placeholder="Leave blank to auto-generate",
        help="A short identifier stored in this session only.",
    )

    upload_mode = st.radio(
        "Upload mode",
        options=["Individual CSVs", "ZIP archive"],
        horizontal=True,
    )

    upload_result = None

    if upload_mode == "Individual CSVs":
        c1, c2, c3 = st.columns(3)
        with c1:
            imu_file = st.file_uploader("IMU CSV", type=["csv"], key="imu_up")
        with c2:
            att_file = st.file_uploader("ATT CSV", type=["csv"], key="att_up")
        with c3:
            baro_file = st.file_uploader("BARO CSV", type=["csv"], key="baro_up")

        if st.button("Validate upload", type="primary", width='stretch'):
            files = {}
            names = {}
            if imu_file:
                files["IMU"] = imu_file.getvalue()
                names["IMU"] = imu_file.name
            if att_file:
                files["ATT"] = att_file.getvalue()
                names["ATT"] = att_file.name
            if baro_file:
                files["BARO"] = baro_file.getvalue()
                names["BARO"] = baro_file.name

            if not files:
                st.error("Select at least one CSV file (all three are required).")
            else:
                with st.spinner("Validating telemetry…"):
                    upload_result = load_individual_csvs(files, filenames=names)

    else:
        zip_file = st.file_uploader("ZIP archive", type=["zip"], key="zip_up")
        st.caption(
            "Archive must contain CSVs named like IMU.csv, ATT.csv, BARO.csv "
            "(folders and common aliases are supported)."
        )
        if st.button("Validate ZIP", type="primary", width='stretch'):
            if not zip_file:
                st.error("Choose a ZIP archive first.")
            else:
                with st.spinner("Extracting and validating…"):
                    upload_result = load_zip_archive(
                        zip_file.getvalue(),
                        archive_name=zip_file.name,
                    )

    if upload_result is not None:
        for err in upload_result.all_errors:
            st.error(err)
        for warn in upload_result.all_warnings:
            st.warning(warn)

        if upload_result.ok and upload_result.validation is not None:
            fid = store_upload(
                st.session_state,
                upload_result,
                flight_id=flight_id_input or None,
            )
            st.success(
                f"Flight **{fid}** validated and stored in this session. "
                "Raw files are kept in memory only."
            )
            st.page_link(
                "pages/1_Flight_Analysis.py",
                label="Continue to Flight Analysis",
                icon="📊",
            )
            render_validation_summary(upload_result.validation.to_dict())
        elif upload_result.validation is not None:
            st.markdown(
                '<p class="status-err">Validation failed — fix the issues above and retry.</p>',
                unsafe_allow_html=True,
            )
            render_validation_summary(upload_result.validation.to_dict())
        else:
            st.markdown(
                '<p class="status-err">Upload could not be processed.</p>',
                unsafe_allow_html=True,
            )

    elif has_flight(st.session_state):
        st.info("A validated flight is already in this session. Reset from the sidebar to upload another.")
        summary = get_flight_summary(st.session_state)
        if summary and summary.get("validation"):
            render_validation_summary(summary["validation"])


if __name__ == "__main__":
    main()
