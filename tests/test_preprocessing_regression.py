"""Semantics-preservation checks for the timestamp refactor.

1. The shared timebase gives the same aligned frame as the original
   ``uav_model/preprocessing.py::load_stream_df`` path for supported inputs.
2. Stored flights (if present locally) re-score to their saved windows.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.preprocessing_adapter import _import_prep, align_uploaded_flight

ROOT = Path(__file__).resolve().parents[1]


def _legacy_align(frames, model_dir):
    prep = _import_prep(model_dir)
    s = prep.DEFAULT_SCHEMA
    imu = prep.load_stream_df(frames["IMU"], s["IMU"])
    att = prep.load_stream_df(frames["ATT"], s["ATT"])
    baro = prep.load_stream_df(frames["BARO"], s["BARO"])
    return prep.align_flight(imu, att, baro, s, dict(prep.DEFAULT_ALIGN))


def _same(a: pd.DataFrame, b: pd.DataFrame):
    assert list(a.columns) == list(b.columns)
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_clean_flight_identical_to_legacy(fixture_model_dir, long_flight_frames):
    new, _ = align_uploaded_flight(long_flight_frames, fixture_model_dir)
    _same(new, _legacy_align(long_flight_frames, fixture_model_dir))


def test_gapped_flight_identical_to_legacy(fixture_model_dir, gapped_flight_frames):
    new, _ = align_uploaded_flight(gapped_flight_frames, fixture_model_dir)
    _same(new, _legacy_align(gapped_flight_frames, fixture_model_dir))


def test_reordered_and_missing_timestamps_identical_to_legacy(fixture_model_dir, long_flight_frames):
    f = {k: v.copy() for k, v in long_flight_frames.items()}
    imu = f["IMU"]
    imu.loc[[7, 8], "TimeUS"] = np.nan
    row20, row21 = imu.iloc[20].copy(), imu.iloc[21].copy()
    imu.iloc[20], imu.iloc[21] = row21, row20  # small reorder
    new, _ = align_uploaded_flight(f, fixture_model_dir)
    _same(new, _legacy_align(f, fixture_model_dir))


def test_adjacent_duplicates_identical_to_legacy_when_rows_identical(fixture_model_dir, long_flight_frames):
    f = {k: v.copy() for k, v in long_flight_frames.items()}
    imu = f["IMU"]
    f["IMU"] = pd.concat([imu.iloc[:13], imu.iloc[10:13], imu.iloc[13:]], ignore_index=True)
    new, _ = align_uploaded_flight(f, fixture_model_dir)
    _same(new, _legacy_align(f, fixture_model_dir))


def _load_verify_module():
    spec = importlib.util.spec_from_file_location("verify_stored", ROOT / "scripts" / "verify_stored_flights.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_stored_flights_reproduce_saved_windows():
    from src.artifact_loader import missing_artifacts

    data_dir = ROOT / "data" / "flights"
    flights = (
        [d for d in data_dir.iterdir() if (d / "windows.parquet").is_file()]
        if data_dir.is_dir()
        else []
    )
    if not flights or missing_artifacts():
        pytest.skip("no stored flights / production artifacts in this checkout")
    from backend.store import FlightStore

    mod = _load_verify_module()
    store = FlightStore(data_dir)
    for d in flights:
        r = mod.compare_flight(store, d.name)
        assert r["reproduced"], r
