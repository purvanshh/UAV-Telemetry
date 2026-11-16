"""Synthetic ARJUNA flights (ULog bytes) with planted faults, written with the converter's own ULog writer."""

from __future__ import annotations

import os
import struct
import tempfile
from typing import Dict, Optional

import numpy as np

from src.arjuna.ulog_writer import Topic, ULogWriter, _rec

HZ = 10
T0 = 1_000_000  # log start, microseconds since boot

TOPICS = [
    Topic("arjuna_vehicle_status", [("uint64_t", "timestamp", 1), ("uint8_t", "flight_mode", 1), ("bool", "armed", 1),
                                    ("uint8_t", "landed_state", 1), ("bool", "failsafe_active", 1), ("uint16_t", "failsafe_reason", 1)]),
    Topic("arjuna_attitude", [("uint64_t", "timestamp", 1), ("float", "roll", 1), ("float", "pitch", 1), ("float", "yaw", 1),
                              ("float", "rates", 3)]),
    Topic("arjuna_vibration", [("uint64_t", "timestamp", 1), ("float", "accel_vibe", 3), ("float", "clip_rate", 1)]),
    Topic("arjuna_actuator_outputs", [("uint64_t", "timestamp", 1), ("uint8_t", "motor_count", 1), ("float", "thrust_sp", 1),
                                      ("float", "motor_cmd", 8), ("uint16_t", "pwm_us", 8), ("uint8_t", "mixer_saturated", 1)]),
    Topic("arjuna_battery", [("uint64_t", "timestamp", 1), ("float", "voltage", 1), ("float", "current", 1),
                             ("uint8_t", "cell_count", 1), ("float", "rail_5v", 1)]),
    Topic("arjuna_imu", [("uint64_t", "timestamp", 1), ("float", "gyro", 3), ("float", "accel", 3), ("uint8_t", "status", 1)]),
    Topic("arjuna_controller", [("uint64_t", "timestamp", 1), ("uint8_t", "controller_id", 1), ("uint8_t", "axis", 1),
                                ("float", "setpoint", 1), ("float", "measured", 1), ("float", "error", 1)]),
    Topic("arjuna_flight_event", [("uint64_t", "timestamp", 1), ("uint16_t", "event_code", 1), ("uint8_t", "category", 1),
                                  ("uint8_t", "severity", 1), ("uint8_t", "mode_before", 1), ("uint8_t", "mode_after", 1)]),
    Topic("arjuna_gps", [("uint64_t", "timestamp", 1), ("uint8_t", "fix_type", 1), ("uint8_t", "satellites", 1), ("float", "h_acc", 1)]),
    Topic("arjuna_position", [("uint64_t", "timestamp", 1), ("float", "pos_u", 1), ("float", "vel_u", 1), ("uint8_t", "reset_counter", 1)]),
    Topic("arjuna_baro", [("uint64_t", "timestamp", 1), ("float", "pressure", 1), ("float", "altitude", 1)]),
]


def flight(*, duration_s: float = 120.0, takeoff_s: float = 10.0, landing_s: float = 110.0, vibe: float = 8.0,
           vibe_spike: Optional[tuple] = None, motor_bias: Optional[Dict[int, float]] = None, cell_min: float = 3.7,
           tilt_spike: Optional[float] = None, crash_event_at: Optional[float] = None, dropouts: int = 0,
           info: Optional[Dict[str, str]] = None, seed: int = 0) -> bytes:
    """One 10 Hz flight. ``vibe_spike=(t0, t1, level)``; ``motor_bias={i: +0.15}`` raises motor i's hover command."""
    rng = np.random.default_rng(seed)
    fd, path = tempfile.mkstemp(suffix=".ulg")
    os.close(fd)
    w = ULogWriter(path, T0)
    base_info = {"sys_name": "ARJUNA", "flight_id": "F09999", "vehicle_id": "ARJUNA_TEST", "schema_version": "1.1.0",
                 "log_profile": "standard", "frame_nav": "ENU", "frame_body": "FLU", "sensors_installed": "imu0,gps0"}
    base_info.update(info or {})
    for k, v in base_info.items():
        w.info(k, v)
    for t in TOPICS:
        w.add_topic(t)
    w.write_formats()
    w.param("MOT_PWM_MAX", 2000.0)
    w.begin_data()
    n = int(duration_s * HZ)
    bias = motor_bias or {}
    for k in range(n):
        t = k / HZ
        ts = T0 + int(t * 1e6)
        air = takeoff_s <= t <= landing_s
        w.data("arjuna_vehicle_status", 0, {"timestamp": ts, "flight_mode": 2, "armed": True, "landed_state": 2 if air else 0,
                                            "failsafe_active": False, "failsafe_reason": 0})
        tilt = np.radians(tilt_spike) if tilt_spike and abs(t - 100.0) < 0.5 else 0.0
        bank = np.radians(12.0) * np.sin(2 * np.pi * (t - 20) / 10) if 20 <= t <= 30 else 0.0  # one manoeuvre
        w.data("arjuna_attitude", 0, {"timestamp": ts, "roll": float(bank + rng.normal(0, 0.01)), "pitch": float(tilt + rng.normal(0, 0.01)),
                                      "yaw": 0.0, "rates": [float(rng.normal(0, 0.02)) for _ in range(3)]})
        v = vibe
        if vibe_spike and vibe_spike[0] <= t <= vibe_spike[1]:
            v = vibe_spike[2]
        w.data("arjuna_vibration", 0, {"timestamp": ts, "accel_vibe": [v * 0.6, v * 0.7, v + float(rng.normal(0, 0.3))],
                                       "clip_rate": 0.0})
        hover = 0.35 if air else 0.0
        cmds = [hover * (1 + bias.get(i, 0.0)) + (float(rng.normal(0, 0.003)) if air else 0.0) for i in range(4)] + [0.0] * 4
        w.data("arjuna_actuator_outputs", 0, {"timestamp": ts, "motor_count": 4, "thrust_sp": hover, "motor_cmd": cmds,
                                              "pwm_us": [int(1000 + 1000 * c) for c in cmds[:4]] + [0] * 4, "mixer_saturated": 0})
        frac = (t - takeoff_s) / max(1e-6, landing_s - takeoff_s) if air else (0.0 if t < takeoff_s else 1.0)
        volt = 3 * (4.15 - (4.15 - cell_min) * frac) - (0.3 if air else 0.0)
        w.data("arjuna_battery", 0, {"timestamp": ts, "voltage": volt, "current": 18.0 if air else 0.5, "cell_count": 3,
                                     "rail_5v": 5.05})
        w.data("arjuna_imu", 0, {"timestamp": ts, "gyro": [float(rng.normal(0, 0.02)) for _ in range(3)],
                                 "accel": [float(rng.normal(0, 0.2)), float(rng.normal(0, 0.2)), 9.81 + float(rng.normal(0, 0.2))],
                                 "status": 0})
        for axis in (0, 1):
            e = float(rng.normal(0, 0.01))
            w.data("arjuna_controller", axis, {"timestamp": ts, "controller_id": 1, "axis": axis, "setpoint": 0.0,
                                               "measured": -e, "error": e})
        if k % 2 == 0:
            w.data("arjuna_gps", 0, {"timestamp": ts, "fix_type": 2, "satellites": 14, "h_acc": 0.8})
        h = 10.0 * min(1.0, max(0.0, (t - takeoff_s) / 5.0, 0.0)) if air else 0.0
        w.data("arjuna_position", 0, {"timestamp": ts, "pos_u": h, "vel_u": 0.0, "reset_counter": 0})
        w.data("arjuna_baro", 0, {"timestamp": ts, "pressure": 101325.0 - 12.0 * h + float(rng.normal(0, 0.6)), "altitude": h + float(rng.normal(0, 0.05))})
        if crash_event_at is not None and abs(t - crash_event_at) < 0.05:
            w.data("arjuna_flight_event", 0, {"timestamp": ts, "event_code": 23, "category": 3, "severity": 2,
                                              "mode_before": 2, "mode_after": 2})
            w.log(2, ts, "Crash: Disarming")
        if dropouts and k in (300, 600)[:dropouts]:
            w._write(_rec("O", struct.pack("<H", 40)))
    w.close()
    data = open(path, "rb").read()
    os.unlink(path)
    return data
