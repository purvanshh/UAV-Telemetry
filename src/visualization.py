"""Plotly figures for the Flight Analysis dashboard."""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.event_builder import AnomalyEvent


def score_timeline_figure(
    windows: pd.DataFrame,
    threshold: float,
    events: Optional[List[AnomalyEvent]] = None,
    selected_event_id: Optional[str] = None,
) -> go.Figure:
    """Score vs time with threshold line and event interval shading."""
    fig = go.Figure()
    if windows is None or len(windows) == 0:
        fig.update_layout(title="No scored windows")
        return fig

    fig.add_trace(
        go.Scatter(
            x=windows["t_start"],
            y=windows["anomaly_score"],
            mode="lines+markers",
            name="anomaly score",
            marker=dict(size=5),
            line=dict(width=1.5, color="#5b8def"),
        )
    )

    flagged = windows[windows["is_anomaly"].astype(bool)]
    if len(flagged):
        fig.add_trace(
            go.Scatter(
                x=flagged["t_start"],
                y=flagged["anomaly_score"],
                mode="markers",
                name="flagged",
                marker=dict(size=8, color="#c45c5c", symbol="diamond"),
            )
        )

    t_min = float(windows["t_start"].min())
    t_max = float(windows["t_end"].max())
    fig.add_trace(
        go.Scatter(
            x=[t_min, t_max],
            y=[threshold, threshold],
            mode="lines",
            name=f"threshold ({threshold:.4f})",
            line=dict(color="#c9a227", dash="dash", width=2),
        )
    )

    for ev in events or []:
        color = "rgba(196, 92, 92, 0.28)" if ev.event_id == selected_event_id else "rgba(196, 92, 92, 0.12)"
        fig.add_vrect(
            x0=ev.start_time_s,
            x1=ev.end_time_s,
            fillcolor=color,
            line_width=0,
            annotation_text=ev.event_id if ev.event_id == selected_event_id else "",
            annotation_position="top left",
        )

    fig.update_layout(
        title="Anomaly score over time",
        xaxis_title="time (s)",
        yaxis_title="reconstruction error",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=60, b=40),
        height=360,
        template="plotly_dark",
    )
    return fig


def telemetry_around_event_figure(
    aligned: pd.DataFrame,
    event: AnomalyEvent,
    features: List[str],
    *,
    pad_s: float = 2.0,
    max_traces: int = 6,
) -> go.Figure:
    """Plot selected aligned features around an event window."""
    fig = go.Figure()
    if aligned is None or aligned.empty or "_t_s" not in aligned.columns:
        fig.update_layout(title="No aligned telemetry available")
        return fig

    t0 = event.start_time_s - pad_s
    t1 = event.end_time_s + pad_s
    sub = aligned[(aligned["_t_s"] >= t0) & (aligned["_t_s"] <= t1)]
    if sub.empty:
        fig.update_layout(title="No samples in event neighborhood")
        return fig

    shown = 0
    for feat in features:
        if feat not in sub.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["_t_s"],
                y=sub[feat],
                mode="lines",
                name=feat,
            )
        )
        shown += 1
        if shown >= max_traces:
            break

    fig.add_vrect(
        x0=event.start_time_s,
        x1=event.end_time_s,
        fillcolor="rgba(196, 92, 92, 0.15)",
        line_width=0,
    )
    fig.update_layout(
        title=f"Telemetry around {event.event_id}",
        xaxis_title="time (s)",
        yaxis_title="value",
        height=320,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def per_feature_error_figure(event: AnomalyEvent) -> go.Figure:
    """Bar chart of peak per-feature reconstruction errors for one event."""
    fig = go.Figure()
    if not event.per_feature_peak:
        fig.update_layout(
            title="No per-feature reconstruction errors available",
            height=280,
            template="plotly_dark",
        )
        return fig

    items = sorted(event.per_feature_peak.items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    fig.add_trace(go.Bar(x=names, y=vals, marker_color="#5b8def"))
    fig.update_layout(
        title=f"Peak per-feature recon error · {event.event_id}",
        xaxis_title="feature",
        yaxis_title="MSE",
        height=300,
        template="plotly_dark",
        margin=dict(l=40, r=20, t=60, b=80),
    )
    return fig


def baro_context_figure(
    aligned: pd.DataFrame,
    event: AnomalyEvent,
    *,
    pad_s: float = 2.0,
) -> go.Figure:
    """Optional BARO raw context — not used by the LSTM."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if aligned is None or aligned.empty:
        fig.update_layout(title="No BARO context")
        return fig
    t0 = event.start_time_s - pad_s
    t1 = event.end_time_s + pad_s
    sub = aligned[(aligned["_t_s"] >= t0) & (aligned["_t_s"] <= t1)]
    if "alt" in sub.columns:
        fig.add_trace(
            go.Scatter(x=sub["_t_s"], y=sub["alt"], name="alt (m)", line=dict(color="#6bbf8a")),
            secondary_y=False,
        )
    if "press" in sub.columns:
        fig.add_trace(
            go.Scatter(x=sub["_t_s"], y=sub["press"], name="press (Pa)", line=dict(color="#c97b4a")),
            secondary_y=True,
        )
    fig.add_vrect(
        x0=event.start_time_s,
        x1=event.end_time_s,
        fillcolor="rgba(196, 92, 92, 0.12)",
        line_width=0,
    )
    fig.update_layout(
        title="BARO raw context (excluded from LSTM)",
        height=280,
        template="plotly_dark",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig.update_yaxes(title_text="alt", secondary_y=False)
    fig.update_yaxes(title_text="press", secondary_y=True)
    return fig
