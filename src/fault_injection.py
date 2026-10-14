"""Synthetic feature-offset injections for threshold sensitivity.

Results are synthetic. They are not real-world diagnostic accuracy.
Injections edit a copy of the aligned feature frame; the caller supplies
the detector. Production artifacts are not modified.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.artifact_loader import ModelArtifacts
from src.detector import UAVDetector
from src.window_generator import generate_windows


def inject_feature_offset(
    aligned: pd.DataFrame,
    feature: str,
    t0: float,
    t1: float,
    delta: float,
) -> pd.DataFrame:
    """Add ``delta`` to ``feature`` on ``[t0, t1]``. Other columns are unchanged."""
    if feature not in aligned.columns:
        raise ValueError(f"aligned frame has no feature {feature}")
    if "_t_s" not in aligned.columns:
        raise ValueError("aligned frame missing _t_s")
    out = aligned.copy()
    mask = (out["_t_s"] >= float(t0)) & (out["_t_s"] <= float(t1))
    out.loc[mask, feature] = pd.to_numeric(out[feature], errors="coerce") + float(delta)
    out.attrs["injection_n_samples"] = int(mask.sum())
    return out


def _overlap_mask(meta_df: pd.DataFrame, t0: float, t1: float) -> pd.Series:
    return (meta_df["t_start"] <= t1) & (meta_df["t_end"] >= t0)


def score_aligned(
    aligned: pd.DataFrame,
    artifacts: ModelArtifacts,
    detector: UAVDetector,
) -> pd.DataFrame:
    batch = generate_windows(aligned, artifacts)
    scores, flags, per_feat = detector.score(batch.X)
    rows = []
    for i, meta in enumerate(batch.meta):
        row = {
            "window_index": i,
            "session_id": meta["session_id"],
            "t_start": meta["t_start"],
            "t_end": meta["t_end"],
            "anomaly_score": float(scores[i]),
            "is_anomaly": bool(flags[i]),
        }
        for j, feat in enumerate(artifacts.features):
            row[f"feat_err_{feat}"] = float(per_feat[i, j])
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_injection(
    *,
    baseline_windows: pd.DataFrame,
    injected_windows: pd.DataFrame,
    feature: str,
    t0: float,
    t1: float,
    threshold: float,
) -> Dict[str, Any]:
    """Compare overlapping windows before and after a synthetic offset."""
    base = baseline_windows.copy()
    inj = injected_windows.copy()
    key = ["session_id", "t_start", "t_end"]
    merged = base.merge(inj, on=key, suffixes=("_base", "_inj"))
    if merged.empty:
        return {
            "synthetic": True,
            "feature": feature,
            "detected": False,
            "reason": "no windows matched on session/time",
        }
    overlap = _overlap_mask(merged, t0, t1)
    outside = ~overlap
    err_col_base = f"feat_err_{feature}_base"
    err_col_inj = f"feat_err_{feature}_inj"
    overlap_rows = merged[overlap]
    detected = bool((overlap_rows["anomaly_score_inj"] > threshold).any()) if len(overlap_rows) else False
    outside_flags = (
        int((merged.loc[outside, "anomaly_score_inj"] > threshold).sum())
        if outside.any()
        else 0
    )
    intended_increased = None
    if len(overlap_rows) and err_col_base in overlap_rows and err_col_inj in overlap_rows:
        intended_increased = bool(
            float(overlap_rows[err_col_inj].mean())
            > float(overlap_rows[err_col_base].mean())
        )
    return {
        "synthetic": True,
        "feature": feature,
        "t0": float(t0),
        "t1": float(t1),
        "threshold": float(threshold),
        "n_overlap_windows": int(overlap.sum()),
        "overlap_score_max_base": (
            float(overlap_rows["anomaly_score_base"].max()) if len(overlap_rows) else None
        ),
        "overlap_score_max_injected": (
            float(overlap_rows["anomaly_score_inj"].max()) if len(overlap_rows) else None
        ),
        "detected_at_threshold": detected,
        "outside_windows_flagged": outside_flags,
        "intended_feature_error_increased": intended_increased,
        "label": "synthetic_not_real_world_accuracy",
    }


def injection_grid(
    *,
    aligned: pd.DataFrame,
    artifacts: ModelArtifacts,
    detector: UAVDetector,
    baseline_windows: pd.DataFrame,
    features: Sequence[str],
    magnitudes_in_std: Sequence[float],
    durations_s: Sequence[float],
    anchors: Dict[str, float],
    threshold: float,
) -> List[Dict[str, Any]]:
    """Run the synthetic grid. ``anchors`` maps a timing name to a start time."""
    std = {
        feat: float(artifacts.scaler["std"][artifacts.features.index(feat)])
        for feat in features
    }
    rows: List[Dict[str, Any]] = []
    for timing, t_anchor in anchors.items():
        for feature in features:
            for k_std in magnitudes_in_std:
                delta = float(k_std) * std[feature]
                for duration in durations_s:
                    t0 = float(t_anchor)
                    t1 = t0 + float(duration)
                    injected = inject_feature_offset(aligned, feature, t0, t1, delta)
                    n_touched = int(injected.attrs.get("injection_n_samples", 0))
                    scored = score_aligned(injected, artifacts, detector)
                    summary = evaluate_injection(
                        baseline_windows=baseline_windows,
                        injected_windows=scored,
                        feature=feature,
                        t0=t0,
                        t1=t1,
                        threshold=threshold,
                    )
                    summary.update(
                        {
                            "timing": timing,
                            "magnitude_std": float(k_std),
                            "delta": delta,
                            "duration_s": float(duration),
                            "n_samples_modified": n_touched,
                        }
                    )
                    rows.append(summary)
    return rows
