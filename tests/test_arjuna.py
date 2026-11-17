"""ARJUNA layer: reading, integrity, compatibility, rule checks and physics on synthetic logs with planted faults."""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest

from src.arjuna.analysis import analyse
from src.arjuna.checks import FAULT_CODES
from src.arjuna.ulog import ArjunaFormatError, is_ulog, load_flight
from tests.fixtures.arjuna_synth import flight


def codes(a):
    return {f["code"] for f in a["findings"]}


@pytest.fixture(scope="module")
def healthy():
    return flight()


def test_healthy_flight_reads_verifies_and_has_no_findings(healthy):
    fl = load_flight(healthy, "flight.ulg")
    assert fl.compatibility["ok"] and fl.integrity["result"] == "verified_up_to"
    assert fl.integrity["blocks_checked"] == fl.integrity["block_count"] >= 1
    assert fl.air_intervals()[0][0] == pytest.approx(10.0, abs=0.11)
    a = analyse(fl)
    assert a["classification"]["kind"] == "flight" and a["findings"] == []
    assert a["verdict"]["worst"] is None
    assert a["physics"]["hover_balance"]["status"] == "computed" and not a["physics"]["hover_balance"]["flag"]
    # every fault code is accounted for: evaluated, or not evaluable with a reason
    for c in FAULT_CODES:
        cov = a["coverage"][c]
        assert cov["status"] == "evaluated" or cov["reasons"], c
    assert a["coverage"]["TIMING"]["status"] == "not_evaluable" and "arjuna_fpga_timing" in a["coverage"]["TIMING"]["reasons"][0]


def test_tampering_is_detected(healthy):
    b = bytearray(healthy)
    b[len(b) // 2] ^= 0xFF
    assert load_flight(bytes(b), "x.ulg").integrity["result"] == "modified"


@pytest.mark.parametrize("key,value", [("schema_version", "2.0.0"), ("frame_nav", "NED"), ("frame_body", "FRD"),
                                       ("log_profile", "fast")])
def test_unknown_schema_or_frames_are_refused_not_guessed(key, value):
    fl = load_flight(flight(info={key: value}), "x.ulg")
    assert not fl.compatibility["ok"] and any(key in p for p in fl.compatibility["problems"])
    a = analyse(fl)
    assert a["refused"] and a["findings"] == []


def test_not_a_ulog_is_rejected():
    assert not is_ulog(b"t_ms,gx\n1,2\n")
    with pytest.raises(ArjunaFormatError):
        load_flight(b"hello world, definitely not a log", "x.ulg")


def test_zipped_flight_folder_reads_sidecars_and_keeps_labels_apart(healthy):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("F09999_2026-11-16/flight.ulg", healthy)
        z.writestr("F09999_2026-11-16/metadata.json", json.dumps({"flight_id": "F09999", "source": {"format": "test"}}))
        z.writestr("F09999_2026-11-16/labels.json", json.dumps({"reviewed": True, "labels": [
            {"code": "VIB_HIGH", "start_us": 30_000_000, "end_us": 40_000_000, "severity": "warning"}]}))
    raw = buf.getvalue()
    assert is_ulog(raw)
    a = analyse(load_flight(raw, "folder.zip"))
    assert a["integrity"]["converted_log"] is True
    assert a["engineer_labels"]["labels"][0]["code"] == "VIB_HIGH"
    assert a["engineer_labels"]["labels"][0]["t_start"] == pytest.approx(29.0)  # relative to the log start


def test_planted_vibration_is_found_with_its_window():
    a = analyse(load_flight(flight(vibe_spike=(50, 60, 75.0)), "x.ulg"))
    v = [f for f in a["findings"] if f["code"] == "VIB_HIGH"]
    assert v and v[0]["severity"] == "critical"
    assert 49 <= v[0]["t_start"] <= 51 and 59 <= v[0]["t_end"] <= 61
    assert v[0]["evidence"][0]["topic"] == "arjuna_vibration" and v[0]["evidence"][0]["value"] >= 60


def test_planted_low_battery_is_critical():
    a = analyse(load_flight(flight(cell_min=2.9), "x.ulg"))
    p = [f for f in a["findings"] if f["code"] == "POWER"]
    assert p and p[0]["severity"] == "critical" and a["verdict"]["worst"] == "critical"


def test_hover_imbalance_names_the_motor_and_reads_the_direction():
    a = analyse(load_flight(flight(motor_bias={2: 0.18}), "x.ulg"))
    m = [f for f in a["findings"] if f["code"] == "MOTOR_ESC"]
    assert m and "M3" in m[0]["title"] and "works harder" in m[0]["title"] and m[0]["window"] == "whole_flight"
    hb = a["physics"]["hover_balance"]
    assert hb["worst_motor"].startswith("M3") and "gives less thrust per command" in hb["interpretation"]
    low = analyse(load_flight(flight(motor_bias={0: -0.2}), "x.ulg"))["physics"]["hover_balance"]
    assert low["worst_motor"].startswith("M1") and "gives more thrust per command" in low["interpretation"]


def test_crash_event_tilt_and_dropouts():
    a = analyse(load_flight(flight(tilt_spike=75.0, crash_event_at=100.2, dropouts=2), "x.ulg"))
    assert {"CRASH", "LOGGING"} <= codes(a)
    crash = [f for f in a["findings"] if f["code"] == "CRASH"]
    assert {f["detector"] for f in crash} == {"event_log", "rule:tilt"}
    assert any(m["level"] == "critical" for m in a["timeline"]["important_messages"])


def test_short_flight_is_analysed_as_a_flight():
    a = analyse(load_flight(flight(takeoff_s=50, landing_s=55), "x.ulg"))
    assert a["classification"]["kind"] == "flight"
    assert a["classification"]["min_airborne_s"] == 0


def test_missing_values_are_nan_never_zero():
    fl = load_flight(flight(), "x.ulg")
    gps = fl.get("arjuna_gps")
    assert np.isfinite(gps["satellites"]).all()  # real values survive
    assert fl.get("arjuna_flight_event") is None  # topic never written: absent, not zero-filled


# ---- pilot briefing -------------------------------------------------------------------------------------------------

def test_pilot_view_is_plain_and_highlights_what_to_do():
    import re
    from src.arjuna.pilot import pilot_view
    v = pilot_view(analyse(load_flight(flight(tilt_spike=75.0, crash_event_at=100.2, motor_bias={2: 0.18}), "x.ulg")))
    assert v["status"] == "critical" and v["status_text"].startswith("Do not fly")
    assert v["problems"][0]["severity"] == "critical" and v["problems"][0]["name"] == "Crash or loss of control"
    assert len({p["code"] for p in v["problems"]}) == len(v["problems"])  # one card per kind of problem
    assert any("M3 front-left" in c for c in v["checklist"])
    text = " ".join([v["status_text"], v["status_detail"], *v["checklist"],
                     *[p[k] for p in v["problems"] for k in ("headline", "meaning", "action")]])
    assert not re.search(r"arjuna_|\w+\[\d\]|rule:|physics:|RMS|p99", text)
    assert "battery" not in v["not_checked"]  # the synthetic flight logs its battery
    clear = pilot_view(analyse(load_flight(flight(), "x.ulg")))
    assert clear["status"] == "clear" and clear["problems"] == [] and "not a guarantee" in clear["status_detail"]
