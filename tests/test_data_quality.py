"""Coverage accounting, window accounting, provenance and metadata propagation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_quality import account_windows
from src.inference_service import analyze_flight, clear_caches
from src.preprocessing_adapter import align_uploaded_flight_detailed
from src.window_generator import generate_windows
from src.artifact_loader import load_artifacts


def _run(model_dir, frames, **kw):
    clear_caches()
    return analyze_flight(frames, flight_id="dq", model_dir=str(model_dir), **kw)


def test_window_accounting_matches_batch_exactly(fixture_model_dir, gapped_flight_frames):
    arts = load_artifacts(fixture_model_dir)
    detail = align_uploaded_flight_detailed(gapped_flight_frames, fixture_model_dir)
    batch = generate_windows(detail.aligned, arts, model_dir=fixture_model_dir)
    acc = account_windows(detail.aligned, arts.features, 30, 15, 0.1)
    assert acc["scored"] == batch.n_windows
    assert acc["candidate"] >= acc["scored"]


def test_nan_windows_are_counted_once_with_reason(fixture_model_dir, long_flight_frames):
    arts = load_artifacts(fixture_model_dir)
    detail = align_uploaded_flight_detailed(long_flight_frames, fixture_model_dir)
    df = detail.aligned.copy()
    df.loc[40:44, "gyro_x"] = np.nan  # hits windows starting at 15 and 30 (stride 15)
    df.loc[42, "acc_y"] = np.nan  # same windows, second feature: must not double-count windows
    acc = account_windows(df, arts.features, 30, 15, 0.1)
    batch = generate_windows(df, arts, model_dir=fixture_model_dir)
    assert acc["candidate"] - acc["scored"] == 2
    assert acc["scored"] == batch.n_windows
    assert acc["by_feature"]["gyro_x"] == 2 and acc["by_feature"]["acc_y"] == 2


def test_report_carries_data_quality(fixture_model_dir, gapped_flight_frames):
    res = _run(fixture_model_dir, gapped_flight_frames)
    dq = res.data_quality
    assert res.report.data_quality == dq
    w = dq["windows"]
    assert w["scored"] == res.n_windows
    assert w["discarded"] == w["candidate"] - w["scored"]
    assert dq["alignment"]["session_count"] == 2
    assert dq["alignment"]["grid_samples"] == (
        dq["alignment"]["grid_samples_in_gaps"] + dq["alignment"]["grid_samples_in_sessions"]
    )
    gap_reasons = [e["reason"] for e in dq["excluded_intervals"]]
    assert "source_gap" in gap_reasons
    for s in ("IMU", "ATT", "BARO"):
        b = dq["streams"][s]
        assert b["rows_input"] == b["rows_after_cleaning"] + b["rows_duplicate_timestamp"] + b["rows_missing_timestamp"]
        assert b["gap_count"] == 1


def test_duplicate_rows_are_not_double_counted(fixture_model_dir, long_flight_frames):
    imu = long_flight_frames["IMU"]
    long_flight_frames["IMU"] = pd.concat([imu.iloc[:13], imu.iloc[10:13], imu.iloc[13:]], ignore_index=True)
    res = _run(fixture_model_dir, long_flight_frames)
    b = res.data_quality["streams"]["IMU"]
    assert b["rows_input"] == len(imu) + 3
    assert b["rows_duplicate_timestamp"] == 3
    assert b["rows_after_cleaning"] == len(imu)


def test_good_flight_has_ok_coverage(fixture_model_dir, long_flight_frames):
    res = _run(fixture_model_dir, long_flight_frames)
    assert res.coverage_status == "ok"
    assert res.data_quality["durations"]["analysed_fraction_of_aligned"] > 0.9


def test_insufficient_flight_is_not_a_normal_completion(fixture_model_dir, sample_frames):
    res = _run(fixture_model_dir, sample_frames)
    assert res.status == "insufficient_data" and not res.ok
    assert res.coverage_status == "insufficient_data"
    assert res.report.data_quality["coverage_status"] == "insufficient_data"
    assert res.data_quality["warnings"]


def test_low_coverage_is_flagged_and_reflected_in_no_flag_warning(fixture_model_dir):
    from tests.conftest import _make_att, _make_baro, _make_imu

    # 4 s bursts separated by long gaps -> most of the aligned span is a gap
    parts = {"IMU": [], "ATT": [], "BARO": []}
    for k in range(4):
        t0 = 1_000_000 + k * 30_000_000
        parts["IMU"].append(_make_imu(n=120, t0_us=t0))
        parts["ATT"].append(_make_att(n=48, t0_us=t0))
        parts["BARO"].append(_make_baro(n=48, t0_us=t0))
    frames = {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}
    res = _run(fixture_model_dir, frames)
    assert res.coverage_status == "low_coverage"
    assert any("Low coverage" in w for w in res.warnings)
    if res.n_flagged == 0:
        assert any("unscored portion" in w for w in res.warnings)


def test_feature_provenance_table(fixture_model_dir, long_flight_frames):
    res = _run(fixture_model_dir, long_flight_frames, flight_meta={
        "filenames": {"IMU": "a_IMU.csv", "ATT": "a_ATT.csv", "BARO": "a_BARO.csv"},
        "file_sha256": {"IMU": "x" * 64, "ATT": "y" * 64, "BARO": "z" * 64},
        "source": "individual", "created_at": "2026-01-01T00:00:00+00:00",
    })
    rows = {r["feature"]: r for r in res.provenance["features"]}
    assert len(rows) == 17
    assert rows["gyro_x"]["unit"] == "rad/s" and rows["gyro_x"]["kind"] == "raw"
    assert rows["yaw_sin"]["kind"] == "derived" and rows["yaw_sin"]["source_fields"] == ["Yaw"]
    assert rows["roll_res"]["source_fields"] == ["Roll", "DesRoll"]
    assert rows["alt"]["used_by_model"] is False and "exclusion_reason" in rows["alt"]
    used = [r["feature"] for r in res.provenance["features"] if r["used_by_model"]]
    assert used == res.features  # same order as the model input
    assert rows["gyro_x"]["scaler_std"] > 0 and rows["gyro_x"]["standardized"]
    # metadata propagation to report
    src = res.report.provenance["source"]
    assert src["filenames"]["IMU"] == "a_IMU.csv" and src["file_sha256"]["ATT"] == "y" * 64
    assert res.report.provenance["timestamps"]["IMU"]["unit"] == "us"
    assert res.report.provenance["deployed"]["input_shape"] == [30, 12]
    assert "checkpoint" in res.report.provenance["artifacts"]
    assert res.data_quality["streams"]["IMU"]["filename"] == "a_IMU.csv"


def test_event_times_map_to_source_clock(fixture_model_dir, long_flight_frames):
    res = _run(fixture_model_dir, long_flight_frames)
    imu_t = long_flight_frames["IMU"]["TimeUS"].to_numpy() / 1e6
    w = res.windows
    assert w["t_start"].min() >= imu_t[0] - 1e-9 and w["t_end"].max() <= imu_t[-1] + 1e-9
    tl = res.data_quality["timeline"]
    assert tl["source_start_s"] == pytest.approx(imu_t[0])
    assert "TimeUS" in tl["time_reference"]
