"""Central telemetry schema: recognition, aliases, unknown columns and unit honesty."""

from __future__ import annotations

import json

import pytest

from src.telemetry_schema import load_schemas, match_schema, resolve_time_header


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("UAV_PROFILE_CONFIRMATIONS", str(tmp_path / "none.json"))
    monkeypatch.setenv("UAV_PROFILES_FILE", str(tmp_path / "none.json"))


def test_schema_is_loadable_and_declares_verified_units():
    schemas = load_schemas()
    assert schemas and schemas[0].id == "uav_flt_csv"
    ds = schemas[0]
    assert ds.time.column == "t_ms" and ds.time.unit == "ms" and ds.time.unit_status == "verified"
    assert ds.columns["roll"].unit == "deg" and ds.columns["roll"].unit_status == "verified"
    assert ds.columns["gyro_x"].unit == "deg/s"
    assert ds.columns["acc_x"].unit == "m/s^2"
    # Undocumented units are never applied.
    assert ds.columns["motor_1"].display_unit is None
    assert ds.columns["throttle_cmd"].display_unit is None
    assert ds.columns["alt_est"].display_unit is None


def test_matching_is_order_and_case_insensitive_and_reports_unknown_columns():
    headers = ["T_MS", "PHI_DEG", "theta_deg", "ax", "ay", "az", "gx", "gy", "gz", "mystery"]
    match = match_schema(headers)
    assert match.applies and match.schema.id == "uav_flt_csv"
    assert resolve_time_header(headers, match) == "T_MS"
    assert match.unmapped == ["mystery"]
    assert match.mapping["PHI_DEG"].canonical == "roll"


def test_aliases_and_missing_optional_columns_are_supported():
    headers = ["timestamp_ms", "roll_deg", "pitch_deg", "ax", "ay", "az", "gx", "gy", "gz",
               "throttle", "roll_cmd", "pitch_cmd", "yaw_cmd", "m1"]
    match = match_schema(headers)
    assert match.applies
    assert resolve_time_header(headers, match) == "timestamp_ms"
    canons = {c.canonical for c in match.mapping.values()}
    assert {"roll", "pitch", "throttle_cmd", "motor_1"} <= canons


def test_unknown_layout_does_not_match():
    match = match_schema(["a", "b", "c"])
    assert not match.applies and match.schema is None
    assert match.kind in ("none", "partial")


def test_schema_file_is_valid_json_and_versioned():
    from pathlib import Path

    raw = json.loads((Path(__file__).resolve().parents[1] / "config" / "telemetry_schema.json").read_text())
    assert raw["version"]
    for ds in raw["datasets"]:
        for name, col in ds["columns"].items():
            if col.get("unit_status") == "verified":
                assert col.get("unit"), name
