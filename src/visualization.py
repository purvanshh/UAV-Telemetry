"""Plotly figures for the Flight Analysis dashboard.

Design notes
------------
* Uses a neutral, accessible colour palette compatible with the CSS design system
  (light & dark).  Templates are set to "plotly_white" so charts look correct on
  the default light background; the app CSS custom-property ``--surface`` acts as
  the page background.
* Signals with different physical scales are placed on separate y-axes or in
  separate sub-charts so they are always readable.
* Every chart carries a human-readable axis title that names the signal and unit.
* Flagged intervals are shaded in orange/red (matching ``--attention``).
* The threshold line is labelled with its numeric value so operators can see the
  limit without diving into the technical appendix.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.event_builder import AnomalyEvent

# ── shared style constants ────────────────────────────────────────────────────────
# Keep in sync with globals.css CSS custom properties where possible.
_ACCENT = "#1f4fa3"        # --accent (light)
_FLAGGED_MARKER = "#9a3412"  # --attention (light)
_THRESHOLD_LINE = "#7a5b00"  # --caution (light)
_FLAG_FILL = "rgba(154, 52, 18, 0.14)"
_FLAG_FILL_SELECTED = "rgba(154, 52, 18, 0.30)"
_UNASSESSED_FILL = "rgba(93, 99, 107, 0.10)"
_OK_LINE = "#1d5c3a"       # --ok (light)

_MARGIN = dict(l=48, r=24, t=52, b=44)
_LEGEND = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
_TEMPLATE = "plotly_white"

# Human-readable signal labels and units (subset used for telemetry charts)
_SIGNAL_META: dict[str, dict[str, str | None]] = {
    "gyro_x": {"label": "Gyro X (angular rate)", "unit": "rad/s"},
    "gyro_y": {"label": "Gyro Y (angular rate)", "unit": "rad/s"},
    "gyro_z": {"label": "Gyro Z (angular rate)", "unit": "rad/s"},
    "acc_x":  {"label": "Accel X", "unit": "m/s²"},
    "acc_y":  {"label": "Accel Y", "unit": "m/s²"},
    "acc_z":  {"label": "Accel Z", "unit": "m/s²"},
    "roll":   {"label": "Roll (measured)", "unit": "deg"},
    "pitch":  {"label": "Pitch (measured)", "unit": "deg"},
    "yaw_sin": {"label": "Heading — sine component", "unit": None},
    "yaw_cos": {"label": "Heading — cosine component", "unit": None},
    "roll_res":  {"label": "Roll residual (meas. − cmd.)", "unit": "deg"},
    "pitch_res": {"label": "Pitch residual (meas. − cmd.)", "unit": "deg"},
}


def _axis_label(signal: str) -> str:
    """Return 'Label (unit)' or just 'Label' if unit is None."""
    m = _SIGNAL_META.get(signal)
    if m is None:
        return signal
    unit = m.get("unit")
    label = m.get("label", signal)
    return f"{label} ({unit})" if unit else str(label)


def _display_label(signal: str) -> str:
    """Short name for the legend / hover."""
    m = _SIGNAL_META.get(signal)
    return m.get("label", signal) if m else signal


# ── individual chart functions ─────────────────────────────────────────────────────

def score_timeline_figure(
    windows: pd.DataFrame,
    threshold: float,
    events: Optional[List[AnomalyEvent]] = None,
    selected_event_id: Optional[str] = None,
) -> go.Figure:
    """Anomaly score vs time with threshold line and event-interval shading.

    Operator-readable:
    * Y-axis is labelled 'Reconstruction error' (not internal jargon).
    * Threshold is shown as a dashed line labelled with its exact value.
    * Flagged points are highlighted in orange-red.
    * Events are shaded; the selected event gets a darker shade.
    * Hover text shows the window's time range and score.
    """
    fig = go.Figure()
    if windows is None or len(windows) == 0:
        fig.update_layout(
            title="No scored windows available",
            template=_TEMPLATE,
            height=320,
            margin=_MARGIN,
        )
        return fig

    t_start = windows["t_start"].to_numpy()
    t_end   = windows["t_end"].to_numpy()
    scores  = windows["anomaly_score"].to_numpy()
    flags   = windows["is_anomaly"].astype(bool).to_numpy()

    # ── all windows as a continuous line ───────────────────────────────────────
    hover = [
        f"<b>Score:</b> {s:.5f}<br><b>Time:</b> {t0:.2f}–{t1:.2f} s<br>"
        f"{'<b>Flagged</b>' if f else 'Normal'}"
        for s, t0, t1, f in zip(scores, t_start, t_end, flags)
    ]
    fig.add_trace(
        go.Scatter(
            x=t_start,
            y=scores,
            mode="lines+markers",
            name="Score per window",
            marker=dict(size=4, color=_ACCENT),
            line=dict(width=1.5, color=_ACCENT),
            hovertext=hover,
            hoverinfo="text",
        )
    )

    # ── flagged windows on top ─────────────────────────────────────────────────
    if flags.any():
        fi = flags.nonzero()[0]
        fig.add_trace(
            go.Scatter(
                x=t_start[fi],
                y=scores[fi],
                mode="markers",
                name="Flagged window",
                marker=dict(size=9, color=_FLAGGED_MARKER, symbol="diamond"),
                hovertext=[hover[i] for i in fi],
                hoverinfo="text",
            )
        )

    # ── threshold line ─────────────────────────────────────────────────────────
    t_min = float(windows["t_start"].min())
    t_max = float(windows["t_end"].max())
    fig.add_trace(
        go.Scatter(
            x=[t_min, t_max],
            y=[threshold, threshold],
            mode="lines",
            name=f"Alert limit ({threshold:.4f})",
            line=dict(color=_THRESHOLD_LINE, dash="dash", width=2),
            hoverinfo="skip",
        )
    )

    # ── event shading ─────────────────────────────────────────────────────────
    for ev in events or []:
        is_sel = ev.event_id == selected_event_id
        fig.add_vrect(
            x0=ev.start_time_s,
            x1=ev.end_time_s,
            fillcolor=_FLAG_FILL_SELECTED if is_sel else _FLAG_FILL,
            line_width=0,
            annotation_text=ev.event_id if is_sel else "",
            annotation_position="top left",
            annotation=dict(font_size=11) if is_sel else None,
        )

    fig.update_layout(
        title="Reconstruction error over time",
        xaxis_title="Time (seconds into recording)",
        yaxis_title="Reconstruction error",
        legend=_LEGEND,
        margin=_MARGIN,
        height=360,
        template=_TEMPLATE,
        hovermode="x unified",
    )
    fig.update_xaxes(ticksuffix=" s")
    return fig


def telemetry_around_event_figure(
    aligned: pd.DataFrame,
    event: AnomalyEvent,
    features: List[str],
    *,
    pad_s: float = 3.0,
    max_traces: int = 6,
) -> go.Figure:
    """Plot aligned signals before / during / after an event.

    Signals are split by physical category (attitude vs inertial vs residuals)
    so that gyro (rad/s) and angle (deg) are never forced onto the same y-axis.
    The flagged interval is shaded in orange-red.
    """
    if aligned is None or aligned.empty or "_t_s" not in aligned.columns:
        fig = go.Figure()
        fig.update_layout(
            title="No aligned telemetry available",
            template=_TEMPLATE,
            height=360,
            margin=_MARGIN,
        )
        return fig

    t0 = event.start_time_s - pad_s
    t1 = event.end_time_s + pad_s
    sub = aligned[(aligned["_t_s"] >= t0) & (aligned["_t_s"] <= t1)].copy()
    if sub.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"No samples in the neighbourhood of {event.event_id}",
            template=_TEMPLATE,
            height=360,
            margin=_MARGIN,
        )
        return fig

    # Group features by physical category so each group shares a y-axis
    _GROUPS: dict[str, list[str]] = {
        "gyro":    ["gyro_x", "gyro_y", "gyro_z"],
        "accel":   ["acc_x", "acc_y", "acc_z"],
        "angle":   ["roll", "pitch", "yaw_sin", "yaw_cos"],
        "residual": ["roll_res", "pitch_res"],
    }
    _GROUP_UNITS: dict[str, str] = {
        "gyro": "rad/s", "accel": "m/s²", "angle": "deg / dimensionless", "residual": "deg"
    }
    _COLOURS: dict[str, list[str]] = {
        "gyro":    ["#1f4fa3", "#2e77c8", "#74a9e8"],
        "accel":   ["#1d5c3a", "#2e9a62", "#7dd4a4"],
        "angle":   ["#7a5b00", "#c9a227", "#e8cf7b"],
        "residual": ["#9a3412", "#e05c33"],
    }

    # Which features are available and belong to this event?
    wanted = [f for f in features if f in sub.columns][:max_traces]
    if not wanted:
        fig = go.Figure()
        fig.update_layout(
            title=f"No model features found in telemetry ({event.event_id})",
            template=_TEMPLATE,
            height=360,
            margin=_MARGIN,
        )
        return fig

    # Organise by group in the order they appear in `wanted`
    groups_present: list[tuple[str, list[str]]] = []
    placed: set[str] = set()
    for gname, gfeats in _GROUPS.items():
        members = [f for f in gfeats if f in wanted and f not in placed]
        if members:
            groups_present.append((gname, members))
            placed.update(members)
    ungrouped = [f for f in wanted if f not in placed]
    if ungrouped:
        groups_present.append(("other", ungrouped))

    n_rows = len(groups_present)
    if n_rows == 0:
        n_rows = 1
    row_heights = [1.0 / n_rows] * n_rows

    y_titles = [
        _GROUP_UNITS.get(gname, "value") for gname, _ in groups_present
    ]
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[" + ".join(_display_label(f) for f in fs) for _, fs in groups_present],
        row_heights=row_heights,
        vertical_spacing=0.06,
    )

    t_rel = sub["_t_s"].to_numpy()
    for row_i, (gname, gfeats) in enumerate(groups_present, start=1):
        colours = _COLOURS.get(gname, ["#1f4fa3", "#2e77c8", "#74a9e8", "#a0c8f0"])
        for ci, feat in enumerate(gfeats):
            vals = sub[feat].to_numpy(dtype=float)
            fig.add_trace(
                go.Scatter(
                    x=t_rel,
                    y=vals,
                    mode="lines",
                    name=_display_label(feat),
                    line=dict(width=1.8, color=colours[ci % len(colours)]),
                    legendgroup=gname,
                    hovertemplate=f"<b>{_display_label(feat)}</b>: %{{y:.4f}}<br>Time: %{{x:.2f}} s<extra></extra>",
                ),
                row=row_i,
                col=1,
            )
        # Shade the flagged interval on each row
        fig.add_vrect(
            x0=event.start_time_s,
            x1=event.end_time_s,
            fillcolor=_FLAG_FILL,
            line_width=0,
            annotation_text="flagged" if row_i == 1 else "",
            annotation_position="top left",
            annotation=dict(font_size=10) if row_i == 1 else None,
            row=row_i,
            col=1,
        )
        fig.update_yaxes(title_text=y_titles[row_i - 1], row=row_i, col=1, tickfont_size=10)

    fig.update_xaxes(
        title_text="Time (seconds into recording)",
        ticksuffix=" s",
        row=n_rows,
        col=1,
    )
    fig.update_layout(
        title=f"Signals before, during and after {event.event_id}",
        height=max(280, 180 * n_rows),
        template=_TEMPLATE,
        legend=_LEGEND,
        margin=_MARGIN,
        hovermode="x unified",
    )
    return fig


def per_feature_error_figure(event: AnomalyEvent) -> go.Figure:
    """Bar chart of peak per-feature reconstruction errors.

    Bars are sorted highest first so the most anomalous signal is immediately
    visible.  Human-readable signal labels are used on the x-axis.
    """
    fig = go.Figure()
    if not event.per_feature_peak:
        fig.update_layout(
            title="Per-feature reconstruction errors not available",
            height=280,
            template=_TEMPLATE,
            margin=_MARGIN,
        )
        return fig

    items = sorted(event.per_feature_peak.items(), key=lambda kv: kv[1], reverse=True)
    raw_names  = [k for k, _ in items]
    disp_names = [_display_label(k) for k in raw_names]
    vals = [v for _, v in items]
    hover = [
        f"<b>{dn}</b><br>Peak error: {v:.5f}<br>Signal: <i>{rn}</i>"
        for dn, rn, v in zip(disp_names, raw_names, vals)
    ]
    colours = [_FLAGGED_MARKER if v > 0 else _ACCENT for v in vals]
    fig.add_trace(
        go.Bar(
            x=disp_names,
            y=vals,
            marker_color=colours,
            hovertext=hover,
            hoverinfo="text",
        )
    )
    fig.update_layout(
        title=f"Signal contribution to flag — {event.event_id} (highest = most unusual)",
        xaxis_title="Signal",
        yaxis_title="Peak reconstruction error (MSE)",
        height=320,
        template=_TEMPLATE,
        margin={**_MARGIN, "b": 100},
        showlegend=False,
    )
    fig.update_xaxes(tickangle=-35, tickfont_size=11)
    return fig


def baro_context_figure(
    aligned: pd.DataFrame,
    event: AnomalyEvent,
    *,
    pad_s: float = 3.0,
) -> go.Figure:
    """Altitude and pressure context around an event.

    BARO signals are not scored by the LSTM; this chart is purely for
    context, labelled clearly as such.  Altitude and pressure are on
    separate y-axes since their scales differ by ~10 000×.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if aligned is None or aligned.empty:
        fig.update_layout(
            title="No barometer context available",
            template=_TEMPLATE,
            height=260,
            margin=_MARGIN,
        )
        return fig
    t0 = event.start_time_s - pad_s
    t1 = event.end_time_s + pad_s
    sub = aligned[(aligned["_t_s"] >= t0) & (aligned["_t_s"] <= t1)]

    has_alt = "alt" in sub.columns
    has_press = "press" in sub.columns

    if has_alt:
        fig.add_trace(
            go.Scatter(
                x=sub["_t_s"],
                y=sub["alt"],
                name="Altitude",
                line=dict(color=_OK_LINE, width=2),
                hovertemplate="<b>Altitude:</b> %{y:.1f} m<br>Time: %{x:.2f} s<extra></extra>",
            ),
            secondary_y=False,
        )
    if has_press:
        fig.add_trace(
            go.Scatter(
                x=sub["_t_s"],
                y=sub["press"],
                name="Pressure",
                line=dict(color="#c97b4a", width=2),
                hovertemplate="<b>Pressure:</b> %{y:.0f} Pa<br>Time: %{x:.2f} s<extra></extra>",
            ),
            secondary_y=True,
        )

    fig.add_vrect(
        x0=event.start_time_s,
        x1=event.end_time_s,
        fillcolor=_FLAG_FILL,
        line_width=0,
        annotation_text="flagged interval",
        annotation_position="top left",
        annotation_font_size=10,
    )
    fig.update_layout(
        title="Barometer context (altitude and pressure — NOT scored by the anomaly detector)",
        height=280,
        template=_TEMPLATE,
        legend=_LEGEND,
        margin=_MARGIN,
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Time (seconds into recording)", ticksuffix=" s")
    fig.update_yaxes(title_text="Altitude (m)", secondary_y=False)
    fig.update_yaxes(title_text="Pressure (Pa)", secondary_y=True)
    return fig
