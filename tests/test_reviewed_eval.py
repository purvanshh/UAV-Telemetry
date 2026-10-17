"""Reviewed-interval evaluation: labels, matching, false alarms, latency."""

from __future__ import annotations

import pandas as pd
import pytest

from src.reviewed_eval import (
    LabelError, descriptive, evaluate, evaluate_flight, per_feature_error_summary, validate_intervals,
)

THR = 4.0


def _windows(rows):
    # rows: (t_start, score) ; windows are 3 s long, stride 1.5 s
    df = pd.DataFrame({
        "window_index": range(len(rows)), "session_id": 0,
        "t_start": [r[0] for r in rows], "t_end": [r[0] + 2.9 for r in rows],
        "anomaly_score": [r[1] for r in rows],
    })
    df["feat_err_gyro_x"] = df["anomaly_score"] * 2
    df["feat_err_acc_x"] = 0.1
    return df


def _iv(label, a, b, fid="f"):
    return {"flight_id": fid, "t_start_s": a, "t_end_s": b, "label": label}


def _labels(*ivs):
    return {"schema": "reviewed_intervals/1", "reviewer": "r", "reviewed_at": "2026-01-01", "intervals": list(ivs)}


def test_validation_rules():
    validate_intervals(_labels(_iv("expected", 0, 5)))
    with pytest.raises(LabelError):
        validate_intervals(_labels(_iv("engine_fault", 0, 5)))  # no fault labels
    with pytest.raises(LabelError):
        validate_intervals(_labels(_iv("expected", 5, 5)))
    with pytest.raises(LabelError):
        validate_intervals(_labels(_iv("expected", 0, 10), _iv("unusual_telemetry", 5, 8)))
    with pytest.raises(LabelError):
        validate_intervals({"schema": "reviewed_intervals/1", "intervals": [_iv("expected", 0, 5)]})  # no provenance
    with pytest.raises(LabelError):
        validate_intervals({"schema": "wrong", "intervals": []})


def test_no_labels_reports_status_not_metrics():
    r = evaluate({"f": _windows([(0, 1.0)])}, _labels(), threshold=THR)
    assert r["status"] == "no_reviewed_labels"
    assert not any(k in r for k in ("event_recall", "false_alarms_per_flight_hour", "precision"))


def test_labels_for_unscored_flight_do_not_produce_metrics():
    r = evaluate({"f": _windows([(0, 1.0)])}, _labels(_iv("unusual_telemetry", 0, 5, fid="other")), threshold=THR)
    assert r["status"] == "no_reviewed_labels" and r["labelled_flights_without_scores"] == ["other"]


def test_overlapping_windows_form_one_event_and_one_detection():
    w = _windows([(0, 1), (1.5, 6), (3.0, 7), (4.5, 8), (6.0, 1), (30, 1)])
    r = evaluate_flight(w, [_iv("unusual_telemetry", 3.0, 5.0)], threshold=THR)
    assert r["n_flagged_windows"] == 3 and r["n_events"] == 1  # not 3 independent events
    assert len(r["detected"]) == 1 and r["missed"] == []
    # first flagged window overlapping [3, 5] starts at 1.5 and ends at 4.4 -> 1.4 s after onset
    assert r["detected"][0]["latency_s"] == pytest.approx(4.4 - 3.0)


def test_missed_interval():
    w = _windows([(0, 1), (1.5, 1), (30, 9)])
    r = evaluate_flight(w, [_iv("unusual_telemetry", 0, 5)], threshold=THR)
    assert len(r["missed"]) == 1 and r["detected"] == []


def test_false_alarm_only_inside_expected_time():
    w = _windows([(0, 9), (20, 9), (40, 9), (60, 9)])  # four separate events
    ivs = [_iv("expected", 0, 10), _iv("data_quality_issue", 19, 25), _iv("uncertain", 39, 45)]
    r = evaluate_flight(w, ivs, threshold=THR)
    assert len(r["false_alarm_events"]) == 1
    assert len(r["events_on_data_quality"]) == 1 and len(r["events_on_uncertain"]) == 1
    assert len(r["unadjudicated_events"]) == 1  # t=60: nobody reviewed it -> never "normal"
    assert r["expected_hours"] == pytest.approx(10 / 3600)


def test_expected_overlapping_unusual_is_not_a_false_alarm():
    w = _windows([(0, 9)])
    r = evaluate_flight(w, [_iv("expected", 0, 20), _iv("unusual_telemetry", 0, 3)], threshold=THR)
    assert r["false_alarm_events"] == [] and len(r["detected"]) == 1


def test_aggregate_metrics():
    w1 = _windows([(0, 9), (50, 1), (100, 9)])
    w2 = _windows([(0, 1), (50, 1)])
    lab = _labels(_iv("unusual_telemetry", 0, 5, "a"), _iv("unusual_telemetry", 49, 52, "a"),
                  _iv("expected", 99, 110, "a"), _iv("expected", 0, 3600, "b"))
    r = evaluate({"a": w1, "b": w2}, lab, threshold=THR)
    assert r["status"] == "ok" and r["reviewed_unusual_intervals"] == 2
    assert r["event_recall"] == 0.5 and len(r["missed_intervals"]) == 1
    assert r["false_alarm_events"] == 1
    assert r["expected_hours_reviewed"] == pytest.approx((11 + 3600) / 3600)
    assert r["false_alarms_per_flight_hour"] == pytest.approx(1 / ((11 + 3600) / 3600))
    assert r["detection_latency_s"]["n"] == 1


def test_threshold_equality_is_not_a_flag():
    r = evaluate_flight(_windows([(0, THR)]), [_iv("unusual_telemetry", 0, 3)], threshold=THR)
    assert r["n_flagged_windows"] == 0 and len(r["missed"]) == 1


def test_per_feature_summary_and_descriptive():
    w = _windows([(0, 1), (2, 9)])
    s = per_feature_error_summary(w, THR)
    assert s["n_flagged"] == 1 and s["features"]["gyro_x"]["mean_flagged"] == 18.0
    assert s["features"]["gyro_x"]["mean_unflagged"] == 2.0
    d = descriptive(w, THR)
    assert d["flagged"] == 1 and "NOT a false-alarm rate" in d["caveat"]
