"""Group flagged windows into anomaly events.

Grouping policy (documented, configurable)
------------------------------------------
1. Keep only windows with ``is_anomaly == True``.
2. Sort by ``session_id``, then ``t_start``.
3. Within one session, merge consecutive flagged windows into the same event
   when their time intervals **overlap** or the gap between the previous
   event's ``t_end`` and the next window's ``t_start`` is ≤ ``max_gap_s``.
4. **Never** merge across different ``session_id`` values (session / source-gap
   boundaries from preprocessing).
5. Event ``start_time_s`` / ``end_time_s`` are ``min(t_start)`` / ``max(t_end)``
   of member windows — not ``n_windows × window_duration``.

Default ``max_gap_s = 1.5`` equals one stride at 10 Hz (15 samples), so
adjacent overlapping flagged windows form one event while isolated spikes
remain separate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

DEFAULT_MAX_GAP_S = 1.5


@dataclass
class AnomalyEvent:
    event_id: str
    session_id: int
    start_time_s: float
    end_time_s: float
    window_indices: List[int] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    per_feature_peak: Dict[str, float] = field(default_factory=dict)
    per_feature_mean: Dict[str, float] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return float(self.end_time_s - self.start_time_s)

    @property
    def flagged_window_count(self) -> int:
        return len(self.window_indices)

    @property
    def peak_score(self) -> float:
        return float(max(self.scores)) if self.scores else float("nan")

    @property
    def mean_flagged_score(self) -> float:
        return float(sum(self.scores) / len(self.scores)) if self.scores else float("nan")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "start_time_s": self.start_time_s,
            "end_time_s": self.end_time_s,
            "duration_s": self.duration_s,
            "flagged_window_count": self.flagged_window_count,
            "peak_score": self.peak_score,
            "mean_flagged_score": self.mean_flagged_score,
            "window_indices": list(self.window_indices),
            "scores": list(self.scores),
            "per_feature_peak": dict(self.per_feature_peak),
            "per_feature_mean": dict(self.per_feature_mean),
        }


def _feature_error_columns(windows: pd.DataFrame) -> List[str]:
    return [c for c in windows.columns if c.startswith("feat_err_")]


def build_events(
    windows: pd.DataFrame,
    *,
    max_gap_s: float = DEFAULT_MAX_GAP_S,
    features: Optional[List[str]] = None,
) -> List[AnomalyEvent]:
    """Build anomaly events from a scored-window table.

    Required columns: ``window_index``, ``session_id``, ``t_start``, ``t_end``,
    ``anomaly_score``, ``is_anomaly``.
    """
    if windows is None or len(windows) == 0:
        return []

    required = {"window_index", "session_id", "t_start", "t_end", "anomaly_score", "is_anomaly"}
    missing = required - set(windows.columns)
    if missing:
        raise ValueError(f"windows missing columns: {sorted(missing)}")

    flagged = windows[windows["is_anomaly"].astype(bool)].copy()
    if flagged.empty:
        return []

    flagged = flagged.sort_values(["session_id", "t_start", "window_index"]).reset_index(drop=True)
    feat_cols = _feature_error_columns(flagged)
    if features:
        # Prefer known feature order when columns exist
        ordered = [f"feat_err_{f}" for f in features if f"feat_err_{f}" in feat_cols]
        feat_cols = ordered or feat_cols

    events: List[AnomalyEvent] = []
    cur: Optional[AnomalyEvent] = None
    event_n = 0

    for _, row in flagged.iterrows():
        sid = int(row["session_id"])
        t0 = float(row["t_start"])
        t1 = float(row["t_end"])
        score = float(row["anomaly_score"])
        widx = int(row["window_index"])

        can_merge = (
            cur is not None
            and cur.session_id == sid
            and t0 <= cur.end_time_s + float(max_gap_s)
        )

        if not can_merge:
            event_n += 1
            cur = AnomalyEvent(
                event_id=f"event-{event_n:03d}",
                session_id=sid,
                start_time_s=t0,
                end_time_s=t1,
                window_indices=[widx],
                scores=[score],
            )
            events.append(cur)
        else:
            assert cur is not None
            cur.end_time_s = max(cur.end_time_s, t1)
            cur.start_time_s = min(cur.start_time_s, t0)
            cur.window_indices.append(widx)
            cur.scores.append(score)

        # Accumulate per-feature errors for this window into the current event
        if feat_cols and cur is not None:
            for col in feat_cols:
                val = row.get(col)
                if val is None or (isinstance(val, float) and val != val):
                    continue
                name = col.replace("feat_err_", "", 1)
                cur.per_feature_peak[name] = max(
                    float(val), cur.per_feature_peak.get(name, float("-inf"))
                )
                # running mean via recount
                prev = cur.per_feature_mean.get(name)
                n = len(cur.window_indices)
                if prev is None or n <= 1:
                    cur.per_feature_mean[name] = float(val)
                else:
                    cur.per_feature_mean[name] = (prev * (n - 1) + float(val)) / n

    # Clean -inf peaks if any feature never set
    for ev in events:
        for k, v in list(ev.per_feature_peak.items()):
            if v == float("-inf"):
                del ev.per_feature_peak[k]

    return events


def events_to_dataframe(events: List[AnomalyEvent], threshold: float) -> pd.DataFrame:
    """Flat event table for CSV / UI."""
    rows = []
    for ev in events:
        peak = ev.peak_score
        rows.append(
            {
                "event_id": ev.event_id,
                "session_id": ev.session_id,
                "start_time_s": ev.start_time_s,
                "end_time_s": ev.end_time_s,
                "duration_s": ev.duration_s,
                "flagged_window_count": ev.flagged_window_count,
                "peak_score": peak,
                "mean_flagged_score": ev.mean_flagged_score,
                "threshold": threshold,
                "peak_over_threshold": (
                    float(peak - threshold) if peak == peak else None
                ),
                "window_indices": ",".join(str(i) for i in ev.window_indices),
            }
        )
    return pd.DataFrame(rows)
