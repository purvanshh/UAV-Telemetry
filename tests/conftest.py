"""Shared fixtures for UAV Sentinel tests."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project_root() -> Path:
    return ROOT


def _make_imu(n: int = 50, t0_us: int = 1_000_000) -> pd.DataFrame:
    t = np.arange(n) * 40_000 + t0_us  # ~25 Hz
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "TimeUS": t,
            "GyrX": rng.normal(0, 0.01, n),
            "GyrY": rng.normal(0, 0.01, n),
            "GyrZ": rng.normal(0, 0.01, n),
            "AccX": rng.normal(0, 0.1, n),
            "AccY": rng.normal(0, 0.1, n),
            "AccZ": rng.normal(-9.8, 0.1, n),
        }
    )


def _make_att(n: int = 20, t0_us: int = 1_000_000) -> pd.DataFrame:
    t = np.arange(n) * 100_000 + t0_us  # ~10 Hz
    return pd.DataFrame(
        {
            "TimeUS": t,
            "Roll": np.linspace(-5, 5, n),
            "Pitch": np.linspace(-2, 2, n),
            "Yaw": np.linspace(0, 90, n),
            "DesRoll": np.zeros(n),
            "DesPitch": np.zeros(n),
            "DesYaw": np.zeros(n),
        }
    )


def _make_baro(n: int = 20, t0_us: int = 1_000_000) -> pd.DataFrame:
    t = np.arange(n) * 100_000 + t0_us
    return pd.DataFrame(
        {
            "TimeUS": t,
            "Alt": np.linspace(0, 10, n),
            "Press": np.full(n, 101325.0),
            "Temp": np.full(n, 22.0),
        }
    )


@pytest.fixture
def sample_frames():
    return {
        "IMU": _make_imu(),
        "ATT": _make_att(),
        "BARO": _make_baro(),
    }


@pytest.fixture
def long_flight_frames():
    """~20 s of overlapping telemetry — enough for multiple 30-sample windows."""
    return {
        "IMU": _make_imu(n=500),
        "ATT": _make_att(n=200),
        "BARO": _make_baro(n=200),
    }


@pytest.fixture
def gapped_flight_frames():
    """Two dense bursts separated by a >1 s gap (session break)."""
    imu_a = _make_imu(n=250, t0_us=1_000_000)
    imu_b = _make_imu(n=250, t0_us=1_000_000 + 10_000_000 + 2_000_000)  # +12 s
    att_a = _make_att(n=100, t0_us=1_000_000)
    att_b = _make_att(n=100, t0_us=1_000_000 + 10_000_000 + 2_000_000)
    baro_a = _make_baro(n=100, t0_us=1_000_000)
    baro_b = _make_baro(n=100, t0_us=1_000_000 + 10_000_000 + 2_000_000)
    return {
        "IMU": pd.concat([imu_a, imu_b], ignore_index=True),
        "ATT": pd.concat([att_a, att_b], ignore_index=True),
        "BARO": pd.concat([baro_a, baro_b], ignore_index=True),
    }


@pytest.fixture
def sample_csv_bytes(sample_frames):
    out = {}
    for stream, df in sample_frames.items():
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        out[stream] = buf.getvalue()
    return out


@pytest.fixture
def sample_zip_bytes(sample_csv_bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for stream, raw in sample_csv_bytes.items():
            zf.writestr(f"{stream}.csv", raw)
    return buf.getvalue()


@pytest.fixture
def nested_zip_bytes(sample_csv_bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("flight_001/IMU/00000001.csv", sample_csv_bytes["IMU"])
        zf.writestr("flight_001/ATT/00000001.csv", sample_csv_bytes["ATT"])
        zf.writestr("flight_001/BARO/00000001.csv", sample_csv_bytes["BARO"])
    return buf.getvalue()


@pytest.fixture(scope="session")
def fixture_model_dir(tmp_path_factory):
    """Build a session-scoped synthetic model dir for Phase 2 tests."""
    from tests.fixtures.build_fixture_model import build_fixture_model

    dest = tmp_path_factory.mktemp("uav_model_fixture")
    build_fixture_model(dest)
    return dest
