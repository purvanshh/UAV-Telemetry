"""Tests for src.upload and session_state."""

from __future__ import annotations

import io
import zipfile

import pandas as pd
import pytest

from src.session_state import (
    get_flight_summary,
    has_flight,
    reset_flight,
    store_upload,
)
from src.upload import (
    classify_stream_from_name,
    load_individual_csvs,
    load_zip_archive,
)


def test_classify_stream_from_name():
    assert classify_stream_from_name("IMU.csv") == "IMU"
    assert classify_stream_from_name("att.csv") == "ATT"
    assert classify_stream_from_name("BARO.CSV") == "BARO"
    assert classify_stream_from_name("flight/IMU/00000001.csv") == "IMU"
    assert classify_stream_from_name("logs/baro_raw.csv") == "BARO"
    assert classify_stream_from_name("readme.csv") is None
    assert classify_stream_from_name("notes.txt") is None


def test_load_individual_valid(sample_csv_bytes):
    result = load_individual_csvs(
        sample_csv_bytes,
        filenames={"IMU": "IMU.csv", "ATT": "ATT.csv", "BARO": "BARO.csv"},
    )
    assert result.ok
    assert set(result.frames) == {"IMU", "ATT", "BARO"}
    assert result.validation is not None
    assert result.validation.ok


def test_load_individual_missing_file(sample_csv_bytes):
    files = {"IMU": sample_csv_bytes["IMU"], "ATT": sample_csv_bytes["ATT"]}
    result = load_individual_csvs(files)
    assert not result.ok
    assert any("BARO" in e for e in result.all_errors)


def test_load_individual_wrong_stream_mapping(sample_csv_bytes):
    # IMU bytes uploaded under ATT slot with an IMU-looking filename
    result = load_individual_csvs(
        {
            "IMU": sample_csv_bytes["ATT"],
            "ATT": sample_csv_bytes["IMU"],
            "BARO": sample_csv_bytes["BARO"],
        },
        filenames={
            "IMU": "ATT.csv",
            "ATT": "IMU.csv",
            "BARO": "BARO.csv",
        },
    )
    assert not result.ok
    assert any("looks like" in e for e in result.errors)


def test_load_zip_flat(sample_zip_bytes):
    result = load_zip_archive(sample_zip_bytes, archive_name="flight.zip")
    assert result.ok
    assert set(result.frames) == {"IMU", "ATT", "BARO"}


def test_load_zip_nested_dirs(nested_zip_bytes):
    result = load_zip_archive(nested_zip_bytes, archive_name="nested.zip")
    assert result.ok
    assert set(result.frames) == {"IMU", "ATT", "BARO"}


def test_load_zip_missing_stream(sample_csv_bytes):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("IMU.csv", sample_csv_bytes["IMU"])
        zf.writestr("ATT.csv", sample_csv_bytes["ATT"])
        # BARO intentionally omitted
    result = load_zip_archive(buf.getvalue())
    assert not result.ok
    assert any("BARO" in e for e in result.errors)


def test_load_zip_ambiguous(sample_csv_bytes):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("IMU.csv", sample_csv_bytes["IMU"])
        zf.writestr("IMU_backup.csv", sample_csv_bytes["IMU"])
        zf.writestr("ATT.csv", sample_csv_bytes["ATT"])
        zf.writestr("BARO.csv", sample_csv_bytes["BARO"])
    result = load_zip_archive(buf.getvalue())
    assert not result.ok
    assert any("Ambiguous" in e for e in result.errors)


def test_load_zip_invalid_archive():
    result = load_zip_archive(b"this is not a zip", archive_name="bad.zip")
    assert not result.ok
    assert any("not a valid ZIP" in e for e in result.errors)


def test_load_zip_no_csv():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("readme.txt", b"hello")
    result = load_zip_archive(buf.getvalue())
    assert not result.ok
    assert any("no CSV" in e for e in result.errors)


def test_malformed_csv_in_upload():
    # Null bytes make pandas refuse to parse as text CSV.
    result = load_individual_csvs(
        {
            "IMU": b"\x00\x01\x02binary-not-csv\xff\xfe",
            "ATT": b"TimeUS,Roll,Pitch,Yaw,DesRoll,DesPitch\n1,0,0,0,0,0\n",
            "BARO": b"TimeUS,Alt,Press,Temp\n1,0,101325,20\n",
        },
        filenames={"IMU": "IMU.csv", "ATT": "ATT.csv", "BARO": "BARO.csv"},
    )
    assert not result.ok
    assert any(
        "Could not parse" in e or "parse" in e.lower() or "empty" in e.lower()
        for e in result.all_errors
    )


def test_session_store_and_reset(sample_csv_bytes):
    upload = load_individual_csvs(
        sample_csv_bytes,
        filenames={"IMU": "IMU.csv", "ATT": "ATT.csv", "BARO": "BARO.csv"},
    )
    assert upload.ok
    session: dict = {}
    reset_flight(session)
    fid = store_upload(session, upload, flight_id="test-flight-1")
    assert fid == "test-flight-1"
    assert has_flight(session)
    summary = get_flight_summary(session)
    assert summary is not None
    assert summary["flight_id"] == "test-flight-1"
    assert summary["status"] == "validated"
    reset_flight(session)
    assert not has_flight(session)
    assert session.get("flight_status") == "idle"


def test_session_rejects_failed_upload(sample_csv_bytes):
    upload = load_individual_csvs({"IMU": sample_csv_bytes["IMU"]})
    session: dict = {}
    with pytest.raises(ValueError):
        store_upload(session, upload)
