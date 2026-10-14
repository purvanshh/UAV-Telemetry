"""Tests for src.validation."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from src.validation import (
    read_csv_bytes,
    resolve_column,
    validate_flight,
    validate_stream,
)


def test_resolve_column_tolerant():
    cols = ["TimeUS", "Gyr X", "AccY"]
    assert resolve_column(cols, "TimeUS") == "TimeUS"
    assert resolve_column(cols, "gyrx") == "Gyr X"
    assert resolve_column(cols, "acc_y") == "AccY"
    assert resolve_column(cols, "missing") is None


def test_valid_imu(sample_frames):
    result = validate_stream(sample_frames["IMU"], "IMU", filename="IMU.csv")
    assert result.ok
    assert result.n_rows == len(sample_frames["IMU"])
    assert result.timestamp_column == "TimeUS"
    assert result.est_hz is not None
    assert abs(result.est_hz - 25.0) < 1.0
    assert not result.errors


def test_valid_flight_different_row_counts(sample_frames):
    # IMU has more rows than ATT/BARO by design
    result = validate_flight(sample_frames)
    assert result.ok
    assert len(result.streams) == 3
    assert any("different row counts" in w for w in result.warnings)


def test_missing_stream(sample_frames):
    frames = {"IMU": sample_frames["IMU"], "ATT": sample_frames["ATT"]}
    result = validate_flight(frames)
    assert not result.ok
    assert any("BARO" in e for e in result.errors)


def test_missing_required_columns(sample_frames):
    bad = sample_frames["IMU"].drop(columns=["GyrX"])
    result = validate_stream(bad, "IMU")
    assert not result.ok
    assert any("GyrX" in e for e in result.errors)


def test_missing_timestamp():
    df = pd.DataFrame({"GyrX": [0.0], "GyrY": [0.0], "GyrZ": [0.0],
                       "AccX": [0.0], "AccY": [0.0], "AccZ": [-9.8]})
    result = validate_stream(df, "IMU")
    assert not result.ok
    assert any("timestamp" in e.lower() for e in result.errors)


def test_empty_dataframe():
    df = pd.DataFrame(columns=["TimeUS", "GyrX", "GyrY", "GyrZ", "AccX", "AccY", "AccZ"])
    result = validate_stream(df, "IMU")
    assert not result.ok


def test_malformed_csv_raises():
    with pytest.raises(ValueError, match="Could not parse"):
        read_csv_bytes(b"not,a\x00valid\xffcsv{{{", filename="bad.csv")


def test_empty_csv_raises():
    with pytest.raises(ValueError, match="empty"):
        read_csv_bytes(b"TimeUS,GyrX\n", filename="empty.csv")


def test_extreme_values_warn_not_reject(sample_frames):
    baro = sample_frames["BARO"].copy()
    baro.loc[0, "Press"] = 201000.0  # known extreme cluster from notebook
    baro.loc[1, "Alt"] = -1.1e6
    result = validate_stream(baro, "BARO")
    assert result.ok
    assert result.range_flags
    assert any("press" in w.lower() or "alt" in w.lower() for w in result.warnings)


def test_duplicate_and_nonmono_timestamps(sample_frames):
    imu = sample_frames["IMU"].copy()
    imu.loc[5, "TimeUS"] = imu.loc[4, "TimeUS"]  # duplicate
    imu.loc[10, "TimeUS"] = imu.loc[8, "TimeUS"] - 1000  # go backwards
    result = validate_stream(imu, "IMU")
    assert result.ok  # warnings only
    assert result.duplicate_timestamps >= 1
    assert result.non_monotonic_timestamps >= 1
