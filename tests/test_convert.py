"""CSV -> ARJUNA ULog conversion: the known FLT format, any other CSV, and the errors."""

from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.arjuna.analysis import analyse
from src.arjuna.convert import ConversionError, convert_csv
from src.arjuna.ulog import load_flight

DATA = Path(__file__).parent / "fixtures" / "data"


def csv_bytes(df: pd.DataFrame) -> bytes:
    b = io.StringIO()
    df.to_csv(b, index=False)
    return b.getvalue().encode()


@pytest.fixture(scope="module")
def flt():
    raw = (DATA / "FLT_0001.CSV").read_bytes()
    ulg, rep = convert_csv(raw, "FLT_0001.CSV")
    return pd.read_csv(DATA / "FLT_0001.CSV"), ulg, rep, load_flight(ulg, "FLT_0001.ulg")


def test_flt_is_recognised_and_every_column_is_accounted_for(flt):
    df, _, rep, fl = flt
    assert rep["format"]["id"] == "uav_flt_csv" and rep["time"]["unit"] == "ms" and rep["time"]["sample_rate_hz"] == 50.0
    mapped = {m["from"] for m in rep["mapped"]}
    dropped = {d["column"] for d in rep["dropped"]}
    assert mapped | dropped | {"t_ms"} == set(df.columns) and not mapped & dropped
    assert {"u1_core", "kill", "aux1"} <= dropped
    assert fl.compatibility["ok"] and fl.integrity["result"] == "verified_up_to"
    assert fl.info["source_format"] == "csv" and fl.info["source_profile"] == "uav_flt_csv"
    assert fl.info["motor_layout"] == "unknown" and fl.info["min_airborne_s"] == "1"


def test_flt_values_are_converted_to_the_standard_units(flt):
    df, _, _, fl = flt
    att, imu, ao = fl.get("arjuna_attitude"), fl.get("arjuna_imu"), fl.get("arjuna_actuator_outputs")
    assert att["roll"].iloc[0] == pytest.approx(math.radians(df["phi_deg"].iloc[0]), abs=1e-6)  # degrees -> radians
    assert imu["gyro[0]"].iloc[5] == pytest.approx(math.radians(df["gx"].iloc[5]), abs=1e-6)  # deg/s -> rad/s
    assert imu["accel[2]"].iloc[0] == pytest.approx(df["az"].iloc[0], abs=1e-5)  # already m/s^2, already FLU
    i = int(df["m1"].idxmax())
    assert ao["motor_cmd[0]"].iloc[i] == pytest.approx((df["m1"].iloc[i] - 1000) / 1000, abs=1e-6)
    assert np.isnan(ao["motor_cmd[4]"]).all()  # unrecorded motors stay NaN, never 0
    assert fl.get("arjuna_rc_input")["channels[2]"].iloc[0] == df["rc_thr"].iloc[0]
    assert fl.params["RCMAP_THROTTLE"] == 3


def test_flt_gets_derived_vibration_and_altitude_loop_and_is_a_flight(flt):
    _, _, rep, fl = flt
    assert {d["record"] for d in rep["derived"]} == {"arjuna_vibration.accel_vibe[0..2]", "arjuna_controller (altitude, z)"}
    ctl = fl.get("arjuna_controller")
    assert set(ctl["controller_id"].unique()) == {2} and set(ctl["axis"].unique()) == {5}
    a = analyse(fl)
    assert a["classification"]["kind"] == "flight" and a["classification"]["limits_from"] == "source conversion"
    assert a["coverage"]["VIB_HIGH"]["status"] == "evaluated"
    assert a["physics"]["hover_balance"]["status"] == "not_computable"  # a 1 s hop has no 5 s of steady hover
    assert a["physics"]["dynamics"]["status"] == "not_computable"


def test_unknown_ardupilot_style_csv_is_mapped_by_name_with_frd_flip_and_unit_inference():
    n = 2000
    t = np.arange(n) * 4000 + 5_000_000             # microseconds, 250 Hz, no unit in the name
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"TimeUS": t, "Roll": rng.normal(2, 1, n), "Pitch": rng.normal(-3, 1, n),
                       "GyrX": rng.normal(0, 0.05, n), "GyrY": rng.normal(0.1, 0.05, n),
                       "AccX": rng.normal(0, 0.01, n), "AccY": rng.normal(0, 0.01, n), "AccZ": rng.normal(-1.0, 0.01, n),
                       "Volt": np.linspace(12.4, 11.1, n), "Mystery": rng.normal(0, 1, n), "Note": ["x"] * n})
    df = df.rename(columns={"TimeUS": "time_us", "Roll": "roll", "Pitch": "pitch", "GyrX": "gyrx", "GyrY": "gyry",
                            "AccX": "accx", "AccY": "accy", "AccZ": "accz", "Volt": "volt"})
    ulg, rep = convert_csv(csv_bytes(df), "export.csv")
    fl = load_flight(ulg, "x.ulg")
    assert rep["format"]["id"] == "generic" and rep["time"]["unit"] == "us"
    assert rep["time"]["sample_rate_hz"] == pytest.approx(250.0)
    text = " ".join(rep["assumptions"])
    assert "imply g" in text and "FRD" in text and "radians" in text  # every inference is written down
    imu, att = fl.get("arjuna_imu"), fl.get("arjuna_attitude")
    assert imu["accel[2]"].median() == pytest.approx(9.80665, rel=0.01)  # g -> m/s^2, then FRD z flipped to FLU
    assert att["pitch"].median() == pytest.approx(3.0, abs=0.2)           # pitch sign flipped; radians kept
    assert fl.get("arjuna_battery")["voltage"].iloc[0] == pytest.approx(12.4, abs=1e-4)
    assert {"Mystery", "Note"} <= {d["column"] for d in rep["dropped"]}


def test_unsorted_and_duplicate_timestamps_are_cleaned_and_reported():
    df = pd.DataFrame({"time_s": [0.0, 0.2, 0.1, 0.1, 0.3, None], "roll_deg": [1, 3, 2, 9, 4, 5]})
    ulg, rep = convert_csv(csv_bytes(df), "x.csv")
    att = load_flight(ulg, "x.ulg").get("arjuna_attitude")
    assert rep["rows_without_valid_time"] == 1 and rep["duplicate_timestamps_dropped"] == 1
    assert np.degrees(att["roll"]).round(3).tolist() == [1.0, 2.0, 3.0, 4.0]


@pytest.mark.parametrize("body,match", [
    (b"a,b\n1,2\n3,4\n", "No time column"),
    (b"time_s,foo\n0,1\n1,2\n", "No column could be mapped"),
    (b"time_s\n0\n", "fewer than two"),
])
def test_unconvertible_files_get_a_clear_error(body, match):
    with pytest.raises(ConversionError, match=match):
        convert_csv(body, "bad.csv")
