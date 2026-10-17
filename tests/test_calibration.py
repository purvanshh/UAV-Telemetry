"""Candidate threshold calibration: deterministic, validation-only, never promotes."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.artifact_loader import load_artifacts
from src.calibration import (
    MIN_CALIBRATION_WINDOWS, calibrate, calibration_flights, quantile_threshold, write_candidate,
)
from src.detector import UAVDetector
from tests.conftest import _make_att, _make_baro, _make_imu


def _long_flight(seed=0, minutes=7.0):
    n_imu = int(minutes * 60 * 25)
    n_10 = int(minutes * 60 * 10)
    rng = np.random.default_rng(seed)
    imu = _make_imu(n=n_imu)
    imu["GyrX"] = rng.normal(0, 0.15, n_imu)
    return {"IMU": imu, "ATT": _make_att(n=n_10), "BARO": _make_baro(n=n_10)}


def _manifest(flights):
    return {"kind": "new_evaluation", "created": "t", "flights": flights}


def _entry(fid, split, eligible=True, available=True):
    return {"flight_id": fid, "split": split, "calibration_eligible": eligible,
            "data_available": available, "content_hash": fid}


@pytest.fixture()
def env(fixture_model_dir):
    arts = load_artifacts(fixture_model_dir)
    return arts, UAVDetector(arts)


def test_quantile_method_is_numpy_linear_on_float32():
    s = np.random.default_rng(0).random(1000).astype(np.float32)
    assert quantile_threshold(s, 0.995) == float(np.quantile(s, 0.995, method="linear"))
    with pytest.raises(Exception):
        quantile_threshold(s, 1.0)


def test_only_eligible_val_flights_selected():
    m = _manifest([_entry("v1", "val"), _entry("v2", "val", eligible=False), _entry("t1", "test"),
                   _entry("tr", "train"), _entry("v3", "val", available=False)])
    assert [f["flight_id"] for f in calibration_flights(m)] == ["v1"]


def test_no_validation_data_is_reported_not_fabricated(env):
    arts, det = env
    res = calibrate(_manifest([_entry("t", "test"), _entry("v", "val", eligible=False)]),
                    artifacts=arts, detector=det, resolve=lambda f: pytest.fail("must not load data"))
    assert res["status"] == "no_validation_data" and res["threshold"] is None and res["promoted"] is False


def test_insufficient_windows_refused(env):
    arts, det = env
    res = calibrate(_manifest([_entry("v", "val")]), artifacts=arts, detector=det,
                    resolve=lambda f: _long_flight(minutes=1.0))
    assert res["status"] == "insufficient_validation_data" and res["threshold"] is None


def test_candidate_matches_manual_quantile_and_records_provenance(env):
    arts, det = env
    flights = {"v1": _long_flight(1), "v2": _long_flight(2)}
    loaded = []

    def resolve(f):
        loaded.append(f["flight_id"])
        return flights[f["flight_id"]]

    man = _manifest([_entry("v1", "val"), _entry("v2", "val"), _entry("test1", "test")])
    res = calibrate(man, artifacts=arts, detector=det, resolve=resolve, quantile=0.99, timestamp="T")
    assert loaded == ["v1", "v2"]  # test flight never opened
    assert res["status"] == "candidate" and res["promoted"] is False
    assert res["n_windows"] >= MIN_CALIBRATION_WINDOWS
    # independent recomputation
    from src.calibration import score_flight_frames

    pooled = np.concatenate([score_flight_frames(flights[k], arts, det, k)["scores32"] for k in ("v1", "v2")])
    assert res["threshold"] == pytest.approx(float(np.quantile(pooled, 0.99, method="linear")))
    assert res["quantile"] == 0.99 and res["flight_ids"] == ["v1", "v2"]
    assert res["calibration_timestamp"] == "T"
    for k in ("checkpoint", "scaler", "preprocessing_py"):
        assert len(res["model_artifacts_sha256"][k]) == 64
    assert res["preprocessing"]["win_samples"] == 30 and res["preprocessing"]["stride_samples"] == 15
    assert res["comparison_operator"] == "score > threshold"
    assert res["score_distribution"]["n"] == res["n_windows"]
    cmp_ = res["comparison_with_deployed"]
    assert cmp_["deployed_threshold"] == arts.threshold and "reproduces_deployed" in cmp_
    assert cmp_["validation_flagged_at_candidate"] <= int(np.ceil(0.01 * res["n_windows"])) + 1


def test_calibration_is_deterministic_and_does_not_touch_deployed_config(env, tmp_path, fixture_model_dir):
    arts, det = env
    cfg = fixture_model_dir / "threshold_config.json"
    before = cfg.read_bytes()
    man = _manifest([_entry("v1", "val")])
    fl = _long_flight(3)
    r1 = calibrate(man, artifacts=arts, detector=det, resolve=lambda f: fl, timestamp="T")
    r2 = calibrate(man, artifacts=arts, detector=det, resolve=lambda f: fl, timestamp="T")
    p1 = write_candidate(r1, tmp_path / "a")
    p2 = write_candidate(r2, tmp_path / "b")
    assert p1.read_bytes() == p2.read_bytes()
    assert p1.name == "threshold_candidate.json"
    assert cfg.read_bytes() == before
    assert json.loads(p1.read_text())["promoted"] is False


def test_strict_inequality_at_threshold():
    s = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    thr = float(s[1])
    assert int((s > np.float32(thr)).sum()) == 1  # equality is not a flag
