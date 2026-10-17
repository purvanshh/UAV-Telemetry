"""Timestamp interpretation: supported units, ambiguity, duplicates, order, gaps."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing_adapter import PreprocessingError, align_uploaded_flight
from src.timebase import normalize_timestamps
from src.validation import validate_flight, validate_stream
from tests.conftest import _make_att, _make_baro, _make_imu


def _df(t, col="TimeUS"):
    return pd.DataFrame({col: t, "v": np.arange(len(t), dtype=float)})


def test_timeus_is_microseconds_explicit():
    clean, rep = normalize_timestamps(_df(np.arange(50) * 40_000 + 5_000_000))
    assert rep.ok and rep.unit == "us" and rep.unit_source == "explicit_field_name"
    assert clean["__t_s"].iloc[0] == pytest.approx(5.0)
    assert rep.est_hz == pytest.approx(25.0)


def test_timestamp_column_uses_schema_default_with_warning():
    clean, rep = normalize_timestamps(_df(np.arange(50) * 40_000, col="Timestamp"))
    assert rep.ok and rep.unit == "us" and rep.unit_source == "schema_default"
    assert any("no unit" in w for w in rep.warnings)


def test_seconds_valued_timestamp_is_rejected_not_reinterpreted():
    # 25 Hz in *seconds*. Read as µs that is 25e6 Hz: ambiguous -> reject.
    clean, rep = normalize_timestamps(_df(np.arange(50) * 0.04, col="Timestamp"))
    assert not rep.ok and rep.status == "unit_ambiguous"
    assert "ambiguous" in rep.errors[0]
    assert clean.empty


def test_milliseconds_valued_timeus_is_rejected():
    clean, rep = normalize_timestamps(_df(np.arange(50) * 40.0 + 1000))  # 25 Hz in ms
    assert rep.status == "unit_ambiguous"


def test_datetime_strings_rejected():
    ts = pd.date_range("2024-01-01", periods=20, freq="40ms").astype(str)
    _clean, rep = normalize_timestamps(_df(ts, col="Timestamp"))
    assert rep.status == "unsupported" and "datetime" in rep.errors[0]


def test_unsupported_column_name():
    _clean, rep = normalize_timestamps(pd.DataFrame({"time": [1, 2, 3], "v": [1, 2, 3]}))
    assert rep.status == "unsupported"


def test_duplicates_counted_once_first_row_kept():
    t = np.arange(20) * 100_000
    t[6] = t[5]
    t[7] = t[5]
    df = _df(t)
    clean, rep = normalize_timestamps(df)
    assert rep.n_rows == 20 and rep.n_duplicates == 2 and rep.n_unique == 18
    assert rep.n_valid == 20 and rep.n_missing == 0
    assert len(clean) == 18
    assert clean.loc[clean["__t_s"] == t[5] / 1e6, "v"].iloc[0] == 5.0  # first kept


def test_minor_nonmonotonic_is_sorted_with_warning():
    t = np.arange(40) * 100_000
    t[10], t[11] = t[11], t[10]
    clean, rep = normalize_timestamps(_df(t))
    assert rep.ok and rep.n_nonmonotonic_steps == 1
    assert (np.diff(clean["__t_s"]) > 0).all()
    assert any("non-monotonic" in w for w in rep.warnings)


def test_clock_reset_is_rejected():
    t = np.concatenate([np.arange(40) * 100_000 + 50_000_000, np.arange(40) * 100_000])
    _clean, rep = normalize_timestamps(_df(t))
    assert rep.status == "time_order_ambiguous"


def test_missing_timestamps_counted_and_dropped():
    t = (np.arange(30) * 100_000).astype(float)
    t[[3, 4]] = np.nan
    clean, rep = normalize_timestamps(_df(t))
    assert rep.n_missing == 2 and rep.n_valid == 28 and len(clean) == 28
    assert rep.n_rows == rep.n_missing + rep.n_duplicates + len(clean)  # no double counting


def test_mostly_missing_timestamps_rejected():
    t = np.full(30, np.nan)
    t[:3] = [0, 100_000, 200_000]
    _clean, rep = normalize_timestamps(_df(t))
    assert not rep.ok


def test_large_gap_listed_before_interpolation():
    t = np.concatenate([np.arange(50) * 100_000, np.arange(50) * 100_000 + 20_000_000])
    _clean, rep = normalize_timestamps(_df(t), gap_threshold_s=0.5)
    assert len(rep.gaps) == 1
    g = rep.gaps[0]
    assert g["duration_s"] == pytest.approx(20.0 - 4.9)
    assert g["start_s"] == pytest.approx(4.9) and g["end_s"] == pytest.approx(20.0)
    assert any("gap" in w for w in rep.warnings)


def test_validation_and_preprocessing_agree_on_units(long_flight_frames):
    """Same frame -> same t_start in validation and aligned output."""
    res = validate_stream(long_flight_frames["IMU"], "IMU")
    aligned, _ = align_uploaded_flight(long_flight_frames)
    assert res.timestamp_unit == "us"
    assert aligned["_t_s"].iloc[0] >= res.t_start - 1e-9
    assert aligned["_t_s"].iloc[0] == pytest.approx(1.0)


def test_validation_rejects_what_preprocessing_would_misread(sample_frames):
    sample_frames["IMU"] = sample_frames["IMU"].rename(columns={"TimeUS": "Timestamp"})
    sample_frames["IMU"]["Timestamp"] = np.arange(50) * 0.04  # seconds
    out = validate_flight(sample_frames)
    assert not out.ok and any("ambiguous" in e for e in out.errors)


def test_alignment_refuses_ambiguous_units(long_flight_frames):
    long_flight_frames["IMU"]["TimeUS"] = long_flight_frames["IMU"]["TimeUS"] / 1e6  # now seconds
    with pytest.raises(PreprocessingError, match="ambiguous"):
        align_uploaded_flight(long_flight_frames)


def test_aligned_timeline_equals_source_clock(long_flight_frames):
    """Grid time is the source clock in seconds: event times map straight back."""
    aligned, _ = align_uploaded_flight(long_flight_frames)
    imu_t = long_flight_frames["IMU"]["TimeUS"].to_numpy() / 1e6
    assert aligned["_t_s"].iloc[0] == pytest.approx(max(imu_t[0], 1.0))
    assert aligned["_t_s"].iloc[-1] < imu_t[-1]
