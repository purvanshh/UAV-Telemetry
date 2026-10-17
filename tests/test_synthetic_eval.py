"""Synthetic injections: determinism, source immutability, recorded outcomes."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from src.artifact_loader import load_artifacts
from src.calibration import score_flight_frames
from src.detector import UAVDetector
from src.synthetic_eval import (
    DEFAULT_LEVELS, INJECTION_TYPES, InjectionSpec, apply_injection, case_seed,
    detection_at_threshold, plan_injection, run_case, sigma_by_feature, summarize,
)
from tests.conftest import _make_att, _make_baro, _make_imu

SIG = {f: 1.0 for f in ["gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z", "roll", "pitch"]}


def _frames(minutes=3.0):
    n = int(minutes * 60 * 25)
    m = int(minutes * 60 * 10)
    imu = _make_imu(n=n)
    imu["GyrX"] = np.random.default_rng(0).normal(0, 0.1, n)
    return {"IMU": imu, "ATT": _make_att(n=m), "BARO": _make_baro(n=m)}


def _spec(t, **kw):
    d = dict(injection_id="x", type=t, columns=["GyrX"], t_start=20.0, t_end=23.0, amplitude_sigma=5.0, seed=1)
    d.update(kw)
    return InjectionSpec(**d)


def test_source_frames_never_modified():
    fr = _frames()
    snap = {k: v.copy(deep=True) for k, v in fr.items()}
    for t in INJECTION_TYPES:
        cols = ["GyrX", "GyrY", "AccZ"] if t == "multi_feature" else ["GyrX"]
        apply_injection(fr, _spec(t, columns=cols), SIG)
    for k in fr:
        pd.testing.assert_frame_equal(fr[k], snap[k])


def test_effects_per_type():
    fr = _frames()
    t = fr["IMU"]["TimeUS"].to_numpy() / 1e6
    iv = (t >= 20) & (t <= 23)
    base = fr["IMU"]["GyrX"].to_numpy()

    out, rec = apply_injection(fr, _spec("spike", n_samples=3), SIG)
    d = out["IMU"]["GyrX"].to_numpy() - base
    assert (d != 0).sum() == 3 and np.allclose(d[d != 0], 5.0) and rec["affected_raw_rows"] == 3

    out, _ = apply_injection(fr, _spec("drift"), SIG)
    d = (out["IMU"]["GyrX"].to_numpy() - base)[iv]
    assert d[0] == pytest.approx(0.0) and d[-1] == pytest.approx(5.0) and (np.diff(d) > 0).all()
    assert (out["IMU"]["GyrX"].to_numpy() - base)[~iv].any() == False  # noqa: E712

    out, _ = apply_injection(fr, _spec("stuck"), SIG)
    assert np.unique(out["IMU"]["GyrX"].to_numpy()[iv]).size == 1

    out, _ = apply_injection(fr, _spec("missing_values"), SIG)
    assert out["IMU"]["GyrX"].isna().sum() == iv.sum() and len(out["IMU"]) == len(fr["IMU"])

    out, rec = apply_injection(fr, _spec("dropout"), SIG)
    assert len(out["IMU"]) == len(fr["IMU"]) - iv.sum() and rec["removed_rows"] == iv.sum()

    out, _ = apply_injection(fr, _spec("oscillation"), SIG)
    d = (out["IMU"]["GyrX"].to_numpy() - base)[iv]
    assert abs(d).max() <= 5.0 + 1e-9 and abs(d).max() > 4.0 and (np.sign(d[:-1]) != np.sign(d[1:])).any()

    out, rec = apply_injection(fr, _spec("multi_feature", columns=["GyrX", "GyrY", "AccZ"]), SIG)
    assert rec["affected_model_features"] == ["acc_z", "gyro_x", "gyro_y"]
    for c in ("GyrX", "GyrY", "AccZ"):
        assert np.allclose((out["IMU"][c].to_numpy() - fr["IMU"][c].to_numpy())[iv], 5.0)


def test_attitude_injection_maps_to_derived_features():
    _, rec = apply_injection(_frames(), _spec("drift", columns=["Roll"]), SIG)
    assert rec["affected_model_features"] == ["roll", "roll_res"] and rec["stream"] == "ATT"


def test_injection_must_stay_in_one_stream_and_have_rows():
    with pytest.raises(ValueError):
        apply_injection(_frames(), _spec("drift", columns=["GyrX", "Roll"]), SIG)
    with pytest.raises(ValueError):
        apply_injection(_frames(), _spec("drift", t_start=9999.0, t_end=10000.0), SIG)


def test_case_seed_deterministic():
    assert case_seed(1, "a", "spike", 0) == case_seed(1, "a", "spike", 0)
    assert case_seed(1, "a", "spike", 0) != case_seed(2, "a", "spike", 0)


@pytest.fixture()
def env(fixture_model_dir):
    import dataclasses

    # The random-weight fixture model scores every window ~8-13; raise its threshold so
    # baseline windows are unflagged (a clean-placement precondition).
    arts = dataclasses.replace(load_artifacts(fixture_model_dir), threshold=14.0)
    det = UAVDetector(arts)
    return arts, det, sigma_by_feature(arts)


def test_plan_is_deterministic_and_placed_on_clean_windows(env):
    arts, det, sig = env
    fr = _frames(minutes=4.0)
    b = score_flight_frames(fr, arts, det, "f")
    w = b["windows"]
    a = plan_injection(w, "drift", DEFAULT_LEVELS["drift"][0], np.random.default_rng(5), 5, "i", arts.threshold)
    c = plan_injection(w, "drift", DEFAULT_LEVELS["drift"][0], np.random.default_rng(5), 5, "i", arts.threshold)
    assert a == c and a is not None
    ov = w[(w["t_start"] <= a.t_end) & (w["t_end"] >= a.t_start)]
    assert (ov["anomaly_score"] <= arts.threshold).all()


def test_run_case_records_everything_and_is_reproducible(env):
    arts, det, sig = env
    fr = _frames(minutes=4.0)
    b = score_flight_frames(fr, arts, det, "f")
    spec = plan_injection(b["windows"], "multi_feature", {"amplitude_sigma": 12.0, "duration_s": 3.0},
                          np.random.default_rng(1), 1, "f:mf", arts.threshold)
    r1, nf = run_case(fr, spec, flight_id="f", base=b, artifacts=arts, detector=det, sigma=sig)
    r2, _ = run_case(fr, spec, flight_id="f", base=b, artifacts=arts, detector=det, sigma=sig)
    assert r1 == r2
    for k in ("source_flight", "type", "columns", "t_start", "t_end", "amplitude_sigma", "seed",
              "affected_raw_rows", "affected_model_features", "pattern_preserved",
              "eligible_for_scoring", "detected", "latency_s", "overlap_windows"):
        assert k in r1
    assert r1["source_flight"] == "f" and r1["seed"] == 1 and r1["eligible_for_scoring"]
    assert r1["windows_overlapping_scored"] >= 1
    assert r1["pattern_preserved"] is True
    assert r1["detected"] == any(w["score"] > arts.threshold for w in r1["overlap_windows"])
    # source untouched
    c = spec.columns[0]
    assert not np.array_equal(nf["IMU"][c].to_numpy(), fr["IMU"][c].to_numpy())


def test_missing_values_make_windows_ineligible(env):
    arts, det, sig = env
    fr = _frames(minutes=4.0)
    b = score_flight_frames(fr, arts, det, "f")
    spec = plan_injection(b["windows"], "missing_values", {"duration_s": 2.0},
                          np.random.default_rng(2), 2, "f:mv", arts.threshold)
    spec.columns = ["GyrX"]
    r, _ = run_case(fr, spec, flight_id="f", base=b, artifacts=arts, detector=det, sigma=sig)
    assert r["eligible_for_scoring"] is False and r["detected"] is False
    assert r["windows_lost_to_injection"] >= 1 and r["pattern_preserved"] is True  # NaNs survive resampling


def test_short_spike_between_grid_points_is_reported_not_preserved(env):
    arts, det, sig = env
    fr = _frames(minutes=4.0)
    b = score_flight_frames(fr, arts, det, "f")
    t = fr["IMU"]["TimeUS"].to_numpy() / 1e6
    # one raw sample ~0.04 s wide, placed between 10 Hz grid ticks
    k = int(np.argmin(np.abs((t % 0.1) - 0.05) + (t < 30) * 9))
    spec = InjectionSpec("f:sp", "spike", ["GyrX"], float(t[k]), float(t[k]) + 0.001, 6.0, n_samples=1, seed=0)
    r, _ = run_case(fr, spec, flight_id="f", base=b, artifacts=arts, detector=det, sigma=sig)
    assert r["preservation_grid_samples_in_interval"] == 0 and r["pattern_preserved"] is False


def _rec(i, elig=True, base=0.1, mx=5.0, ts=10.0, t_end=13.0):
    return {"injection_id": i, "type": "spike", "eligible_for_scoring": elig, "baseline_max_overlap_score": base,
            "max_overlap_score": mx if elig else None, "t_start": ts, "pattern_preserved": True,
            "overlap_windows": [{"t_start": ts - 1, "t_end": t_end, "score": mx}] if elig else []}


def test_detection_accounting_and_latency():
    recs = [_rec("a", mx=5.0), _rec("b", mx=1.0), _rec("c", elig=False), _rec("d", base=3.0, mx=6.0)]
    d = detection_at_threshold(recs, 4.0)
    assert (d["n_injections"], d["n_eligible"], d["n_not_eligible"]) == (4, 3, 1)
    assert d["n_confounded_by_baseline"] == 0 and d["n_evaluable"] == 3 and d["n_detected"] == 2
    assert d["detection_rate_eligible"] == pytest.approx(2 / 3) and d["missed"] == ["b"]
    assert d["latency_s_median"] == pytest.approx(3.0)
    d2 = detection_at_threshold(recs, 2.0)  # baseline of d (3.0) exceeds 2.0 -> confounded
    assert d2["n_confounded_by_baseline"] == 1 and d2["n_evaluable"] == 2
    assert summarize(recs, 4.0)["overall"]["n_detected"] == 2
