"""Offline threshold-sensitivity calculations.

Does not read or write ``threshold_config.json`` and does not select a
production threshold. Validation quantiles are computed only from rows
whose ``split`` is ``val``.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.event_builder import DEFAULT_MAX_GAP_S, build_events

CANDIDATE_QUANTILES = (0.99, 0.995, 0.999)
REQUIRED_SCORE_COLUMNS = (
    "flight_id",
    "split",
    "window_index",
    "session_id",
    "t_start",
    "t_end",
    "anomaly_score",
)


def score_distribution(scores: Iterable[float]) -> Dict[str, Any]:
    arr = np.asarray(list(scores), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {
            "n": 0,
            "min": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "p995": None,
            "p999": None,
            "max": None,
            "mean": None,
        }
    return {
        "n": int(len(arr)),
        "min": float(np.min(arr)),
        "p50": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "p995": float(np.quantile(arr, 0.995)),
        "p999": float(np.quantile(arr, 0.999)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def candidate_thresholds_from_validation(
    scores: Iterable[float],
    quantiles: Sequence[float] = CANDIDATE_QUANTILES,
) -> Dict[str, float]:
    """Quantile thresholds fit on validation scores only."""
    arr = np.asarray(list(scores), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 20:
        raise ValueError(
            f"Need at least 20 finite validation scores to compare quantiles; got {len(arr)}."
        )
    out: Dict[str, float] = {}
    for q in quantiles:
        if not 0.0 < float(q) < 1.0:
            raise ValueError(f"quantile must be in (0, 1); got {q}")
        out[f"q{q:g}"] = float(np.quantile(arr, float(q)))
    return out


def windows_at_threshold(windows: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Copy windows and set ``is_anomaly`` from ``score > threshold``.

    Equality with the threshold is not a flag. The input frame is not mutated.
    """
    out = windows.copy()
    scores = pd.to_numeric(out["anomaly_score"], errors="coerce")
    out["anomaly_score"] = scores
    out["is_anomaly"] = scores > float(threshold)
    out["threshold"] = float(threshold)
    return out


def summarize_threshold(
    windows: pd.DataFrame,
    threshold: float,
    *,
    max_gap_s: float = DEFAULT_MAX_GAP_S,
    features: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Per-flight and aggregate flag/event stats at one threshold."""
    flagged_table = windows_at_threshold(windows, threshold)
    n = int(len(flagged_table))
    n_flagged = int(flagged_table["is_anomaly"].sum()) if n else 0
    per_flight: List[Dict[str, Any]] = []
    n_events = 0
    flights_with_flag = 0
    if "flight_id" not in flagged_table.columns:
        groups = [("all", flagged_table)]
    else:
        groups = list(flagged_table.groupby("flight_id", sort=True))
    for flight_id, group in groups:
        g_flagged = int(group["is_anomaly"].sum())
        events = build_events(group, max_gap_s=max_gap_s, features=features)
        if g_flagged > 0:
            flights_with_flag += 1
        n_events += len(events)
        per_flight.append(
            {
                "flight_id": str(flight_id),
                "n_windows": int(len(group)),
                "n_flagged": g_flagged,
                "flagged_fraction": (g_flagged / len(group)) if len(group) else None,
                "n_events": len(events),
                "score_max": float(group["anomaly_score"].max()) if len(group) else None,
            }
        )
    return {
        "threshold": float(threshold),
        "n_windows": n,
        "n_flagged": n_flagged,
        "flagged_fraction": (n_flagged / n) if n else None,
        "n_flights": len(per_flight),
        "n_flights_with_flag": flights_with_flag,
        "n_events": n_events,
        "per_flight": per_flight,
    }


def compare_unseen_to_reference(
    unseen_scores: Iterable[float],
    reference_scores: Iterable[float],
) -> Dict[str, Any]:
    """Locate an unseen flight's max score inside a reference distribution.

    Does not assign a fault label.
    """
    unseen = np.asarray(list(unseen_scores), dtype=np.float64)
    ref = np.asarray(list(reference_scores), dtype=np.float64)
    unseen = unseen[np.isfinite(unseen)]
    ref = ref[np.isfinite(ref)]
    if len(unseen) == 0 or len(ref) == 0:
        return {"comparable": False, "reason": "empty score array"}
    unseen_max = float(np.max(unseen))
    rank = float(np.mean(ref <= unseen_max))
    return {
        "comparable": True,
        "label": None,
        "unseen_n": int(len(unseen)),
        "unseen_max": unseen_max,
        "unseen_p95": float(np.quantile(unseen, 0.95)),
        "reference_n": int(len(ref)),
        "reference_p50": float(np.quantile(ref, 0.50)),
        "reference_p99": float(np.quantile(ref, 0.99)),
        "reference_p995": float(np.quantile(ref, 0.995)),
        "reference_max": float(np.max(ref)),
        "fraction_of_reference_at_or_below_unseen_max": rank,
        "note": (
            "Proximity of unseen_max to a threshold is not a fault label."
        ),
    }


def split_ids_from_schema(feature_schema: Dict[str, Any]) -> Dict[str, List[str]]:
    raw = feature_schema.get("split") or {}
    return {
        "train": [str(x) for x in raw.get("train") or []],
        "val": [str(x) for x in raw.get("val") or []],
        "test": [str(x) for x in raw.get("test") or []],
    }


def missing_score_corpus(feature_schema: Dict[str, Any]) -> Dict[str, Any]:
    splits = split_ids_from_schema(feature_schema)
    return {
        "available": False,
        "required_file": "data/eval/window_scores.parquet",
        "required_columns": list(REQUIRED_SCORE_COLUMNS),
        "required_split_values": ["val", "test"],
        "val_flight_ids": splits["val"],
        "test_flight_ids": splits["test"],
        "n_val_flights_listed": len(splits["val"]),
        "n_test_flights_listed": len(splits["test"]),
        "must_regenerate": [
            "Per-window reconstruction scores for every validation flight listed in feature_schema.json split.val, produced by this checkpoint and preprocessing (not a different variant).",
            "The same per-window scores for split.test, kept in a separate split column.",
            "The original calibration script is not in this repo. The stored threshold is the 0.995 quantile of validation window scores (threshold_config.json). Regenerating it requires those validation scores; do not recompute the quantile from test flights or from flight 00000122.",
        ],
    }


def load_score_table(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    missing = [c for c in REQUIRED_SCORE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"score table missing columns: {missing}")
    df = df.copy()
    df["split"] = df["split"].astype(str)
    df["flight_id"] = df["flight_id"].astype(str)
    return df


def evaluate_score_table(
    scores: pd.DataFrame,
    *,
    deployed_threshold: float,
    quantiles: Sequence[float] = CANDIDATE_QUANTILES,
) -> Dict[str, Any]:
    """Validation-fit candidate thresholds, then describe test at the deployed threshold only."""
    val = scores[scores["split"] == "val"]
    test = scores[scores["split"] == "test"]
    if val.empty:
        raise ValueError("score table has no split=='val' rows")
    candidates = candidate_thresholds_from_validation(val["anomaly_score"], quantiles)
    # Record the deployed threshold alongside candidates. Do not replace it
    # with a test-set quantile.
    named = {"deployed": float(deployed_threshold), **candidates}
    validation_report = {
        name: summarize_threshold(val, thr) for name, thr in named.items()
    }
    test_report = None
    if not test.empty:
        test_report = {
            "deployed_only": summarize_threshold(test, deployed_threshold),
            "distribution": score_distribution(test["anomaly_score"]),
            "note": "Test flights are not used to choose a threshold.",
        }
    return {
        "validation_distribution": score_distribution(val["anomaly_score"]),
        "candidate_thresholds": named,
        "validation_at_candidates": validation_report,
        "test_at_deployed": test_report,
    }
