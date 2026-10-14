"""Threshold-sensitivity math. Does not touch the production threshold file."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.fault_injection import evaluate_injection, inject_feature_offset
from src.threshold_sensitivity import (
    candidate_thresholds_from_validation,
    compare_unseen_to_reference,
    evaluate_score_table,
    missing_score_corpus,
    score_distribution,
    summarize_threshold,
    windows_at_threshold,
)


def _windows(scores, flight_id="f1", session=0):
    rows = []
    for i, score in enumerate(scores):
        rows.append(
            {
                "flight_id": flight_id,
                "window_index": i,
                "session_id": session,
                "t_start": i * 1.5,
                "t_end": i * 1.5 + 2.9,
                "anomaly_score": score,
                "is_anomaly": False,
            }
        )
    return pd.DataFrame(rows)


def test_candidate_quantiles_use_only_the_supplied_validation_scores():
    val = np.linspace(0.0, 1.0, 1001)
    thr = candidate_thresholds_from_validation(val, (0.99, 0.995, 0.999))
    assert thr["q0.99"] == pytest.approx(0.99, abs=1e-3)
    assert thr["q0.995"] == pytest.approx(0.995, abs=1e-3)
    assert thr["q0.999"] == pytest.approx(0.999, abs=1e-3)
    with pytest.raises(ValueError):
        candidate_thresholds_from_validation([1.0, 2.0], (0.99,))


def test_flags_and_events_at_a_candidate_threshold():
    # Two adjacent flags in one flight, one isolated flag in another, one equal-to-threshold.
    a = _windows([1.0, 5.0, 5.2, 1.0], flight_id="a")
    b = _windows([4.0, 1.0], flight_id="b")
    windows = pd.concat([a, b], ignore_index=True)
    marked = windows_at_threshold(windows, 4.0)
    assert marked.loc[marked["flight_id"] == "b", "is_anomaly"].tolist() == [False, False]
    summary = summarize_threshold(windows, 4.0)
    assert summary["n_windows"] == 6
    assert summary["n_flagged"] == 2
    assert summary["n_flights_with_flag"] == 1
    assert summary["n_events"] == 1
    assert summary["flagged_fraction"] == pytest.approx(2 / 6)


def test_evaluate_score_table_does_not_fit_threshold_on_test():
    val = _windows(np.linspace(0, 10, 200), flight_id="v")
    val["split"] = "val"
    test = _windows([100.0, 100.0, 100.0], flight_id="t")
    test["split"] = "test"
    table = pd.concat([val, test], ignore_index=True)
    report = evaluate_score_table(table, deployed_threshold=4.034964084625244)
    assert report["candidate_thresholds"]["deployed"] == pytest.approx(4.034964084625244)
    # Test scores are 100; a test-fit 0.995 quantile would be ~100, not the val quantile.
    assert report["candidate_thresholds"]["q0.995"] < 20
    assert report["test_at_deployed"]["note"]
    assert report["test_at_deployed"]["deployed_only"]["n_flagged"] == 3


def test_unseen_comparison_does_not_assign_a_fault_label():
    ref = np.linspace(0, 4.0, 500)
    out = compare_unseen_to_reference([0.2, 4.0063], ref)
    assert out["comparable"] is True
    assert out["label"] is None
    assert out["unseen_max"] == pytest.approx(4.0063)
    assert "not a fault label" in out["note"]


def test_missing_corpus_lists_schema_splits():
    schema = {"split": {"val": ["00000007"], "test": ["00000009"], "train": ["00000001"]}}
    missing = missing_score_corpus(schema)
    assert missing["available"] is False
    assert missing["val_flight_ids"] == ["00000007"]
    assert missing["test_flight_ids"] == ["00000009"]
    assert any("00000122" in item or "validation" in item.lower() for item in missing["must_regenerate"])


def test_injection_changes_only_the_intended_feature_and_interval():
    aligned = pd.DataFrame(
        {
            "_t_s": np.arange(0, 10, 0.1),
            "gyro_x": np.zeros(100),
            "roll": np.ones(100),
        }
    )
    out = inject_feature_offset(aligned, "gyro_x", 2.0, 4.0, delta=3.5)
    touched = (out["_t_s"] >= 2.0) & (out["_t_s"] <= 4.0)
    assert np.allclose(out.loc[touched, "gyro_x"], 3.5)
    assert np.allclose(out.loc[~touched, "gyro_x"], 0.0)
    assert np.allclose(out["roll"], 1.0)
    assert np.allclose(aligned["gyro_x"], 0.0)

    base = pd.DataFrame(
        {
            "session_id": [0, 0],
            "t_start": [1.0, 5.0],
            "t_end": [3.9, 7.9],
            "anomaly_score": [0.2, 0.2],
            "feat_err_gyro_x": [0.1, 0.1],
        }
    )
    inj = base.copy()
    inj["anomaly_score"] = [6.0, 0.2]
    inj["feat_err_gyro_x"] = [4.0, 0.1]
    summary = evaluate_injection(
        baseline_windows=base,
        injected_windows=inj,
        feature="gyro_x",
        t0=2.0,
        t1=4.0,
        threshold=4.034964084625244,
    )
    assert summary["synthetic"] is True
    assert summary["detected_at_threshold"] is True
    assert summary["intended_feature_error_increased"] is True
    assert summary["outside_windows_flagged"] == 0
    assert summary["label"] == "synthetic_not_real_world_accuracy"


def test_score_distribution_empty():
    assert score_distribution([])["n"] == 0
