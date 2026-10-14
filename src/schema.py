"""Sensor schema conventions matching the UAV preprocessing notebook.

Column names match the ArduPilot-style IMU / ATT / BARO CSVs used by the
training dataset. Required columns are the minimum set needed for Phase 2
alignment; optional columns are accepted when present.
"""

from __future__ import annotations

from typing import Any, Dict, List

STREAMS: tuple[str, ...] = ("IMU", "ATT", "BARO")

# Timestamp candidates in priority order (notebook SCHEMA["ts_priority"]).
TIMESTAMP_CANDIDATES: List[str] = ["TimeUS", "Timestamp"]

# Canonical feature → source column mapping from the notebook.
STREAM_SCHEMA: Dict[str, Dict[str, Any]] = {
    "IMU": {
        "required_features": {
            "gyro_x": "GyrX",
            "gyro_y": "GyrY",
            "gyro_z": "GyrZ",
            "acc_x": "AccX",
            "acc_y": "AccY",
            "acc_z": "AccZ",
        },
        "units": {
            "gyro_x": "rad/s",
            "gyro_y": "rad/s",
            "gyro_z": "rad/s",
            "acc_x": "m/s^2",
            "acc_y": "m/s^2",
            "acc_z": "m/s^2",
        },
        "range_checks": {
            "gyro_x": (-60.0, 60.0),
            "gyro_y": (-60.0, 60.0),
            "gyro_z": (-60.0, 60.0),
            "acc_x": (-200.0, 200.0),
            "acc_y": (-200.0, 200.0),
            "acc_z": (-200.0, 200.0),
        },
    },
    "ATT": {
        "required_features": {
            "roll": "Roll",
            "pitch": "Pitch",
            "yaw": "Yaw",
            "des_roll": "DesRoll",
            "des_pitch": "DesPitch",
        },
        "optional_features": {
            "des_yaw": "DesYaw",
        },
        "units": {
            "roll": "deg",
            "pitch": "deg",
            "yaw": "deg",
            "des_roll": "deg",
            "des_pitch": "deg",
            "des_yaw": "deg",
        },
        "range_checks": {
            "roll": (-180.0, 180.0),
            "pitch": (-90.0, 90.0),
            "yaw": (-180.0, 360.0),
            "des_roll": (-180.0, 180.0),
            "des_pitch": (-90.0, 90.0),
        },
    },
    "BARO": {
        "required_features": {
            "alt": "Alt",
            "press": "Press",
            "temp": "Temp",
        },
        "optional_features": {
            "alt_amsl": "AltAMSL",
        },
        "units": {
            "alt": "m",
            "alt_amsl": "m",
            "press": "Pa",
            "temp": "degC",
        },
        "range_checks": {
            "alt": (-1000.0, 20000.0),
            "press": (1000.0, 120000.0),
            "temp": (-80.0, 100.0),
        },
    },
}

# Filename tokens used to map uploaded files / ZIP members to streams.
STREAM_FILENAME_HINTS: Dict[str, List[str]] = {
    "IMU": ["imu"],
    "ATT": ["att", "attitude"],
    "BARO": ["baro", "barometer"],
}
