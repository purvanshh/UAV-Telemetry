"""Units and meanings of ARJUNA topic fields (Flight Logging Standard v1.1, "Field definitions").

Angles are stored in radians; :func:`display` converts them to degrees for people and for the agent, so the
language model never converts units itself. Fields not listed here are reported with unit ``None``.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

# topic -> field prefix (before "[i]") -> (unit, meaning)
FIELDS: Dict[str, Dict[str, Tuple[Optional[str], str]]] = {
    "arjuna_vehicle_status": {
        "flight_mode": ("enum", "flight mode"), "armed": (None, "motors armed"), "landed_state": ("enum", "landed state"),
        "failsafe_active": (None, "any failsafe in force"), "failsafe_reason": ("enum", "failsafe reason"),
        "control_source": ("enum", "control source"), "prearm_failures": ("bitflags", "failed pre-arm checks"),
    },
    "arjuna_imu": {
        "gyro": ("rad/s", "converted gyro, body FLU"), "accel": ("m/s^2", "converted acceleration, body FLU"),
        "temperature": ("degC", "IMU temperature"), "clip_count": ("count", "samples at range limit since last record"),
        "error_count": ("count", "bus read errors so far"), "status": ("enum", "IMU status"),
        "sample_seq": ("count", "sample counter; a jump means lost samples"),
        "gyro_raw": ("LSB", "raw gyro"), "accel_raw": ("LSB", "raw accelerometer"),
    },
    "arjuna_imu_filtered": {
        "gyro_bias": ("rad/s", "estimated gyro bias"), "gyro_corrected": ("rad/s", "gyro minus bias"),
        "gyro_notch1": ("rad/s", "after first notch"), "gyro_notch2": ("rad/s", "after second notch"),
        "gyro_filtered": ("rad/s", "final gyro used by controllers"), "accel_filtered": ("m/s^2", "final acceleration"),
        "notch_freq": ("Hz", "notch centre frequency"), "status": ("enum", "filter health"),
    },
    "arjuna_vibration": {
        "accel_vibe": ("m/s^2", "accel vibration level (high-frequency RMS)"), "gyro_vibe": ("rad/s", "gyro vibration level"),
        "clip_rate": ("1/s", "clipped samples per second, all axes"), "peak_freq": ("Hz", "strongest vibration frequency"),
    },
    "arjuna_gps": {
        "latitude": ("deg", "latitude"), "longitude": ("deg", "longitude"), "alt_msl": ("m", "altitude MSL"),
        "vel_e": ("m/s", "velocity east"), "vel_n": ("m/s", "velocity north"), "vel_u": ("m/s", "velocity up"),
        "ground_speed": ("m/s", "horizontal speed"), "course": ("rad", "course over ground"), "heading": ("rad", "antenna heading"),
        "satellites": ("count", "satellites used"), "fix_type": ("enum", "fix type"), "h_acc": ("m", "horizontal accuracy"),
        "v_acc": ("m", "vertical accuracy"), "s_acc": ("m/s", "speed accuracy"), "noise_level": ("receiver units", "RF noise"),
        "jamming_indicator": ("0-255", "jamming estimate"), "spoofing_state": ("enum", "spoofing flag"), "status": ("enum", "GPS health"),
    },
    "arjuna_baro": {
        "pressure": ("Pa", "pressure"), "altitude": ("m", "pressure altitude"), "temperature": ("degC", "baro temperature"),
        "error_count": ("count", "bus read errors"), "status": ("enum", "baro health"),
    },
    "arjuna_mag": {
        "field": ("gauss", "magnetic field, body FLU"), "field_strength": ("gauss", "total field strength"),
        "temperature": ("degC", "mag temperature"), "status": ("enum", "mag health"),
    },
    "arjuna_rangefinder": {
        "distance_raw": ("m", "distance as measured"), "distance": ("m", "distance after filtering"),
        "signal_quality": ("%", "signal quality"), "valid": (None, "reading usable"),
    },
    "arjuna_attitude": {
        "quaternion": (None, "w,x,y,z body FLU to nav ENU"), "roll": ("rad", "roll (+ = right wing down)"),
        "pitch": ("rad", "pitch (+ = nose down)"), "yaw": ("rad", "yaw (+ = nose left)"), "rates": ("rad/s", "body rates"),
        "gyro_bias": ("rad/s", "attitude-filter gyro bias"), "accel_weight": ("0-1", "accelerometer trust"),
        "valid": (None, "attitude usable"), "status": ("enum", "estimator health"),
    },
    "arjuna_position": {
        "pos_e": ("m", "position east of home"), "pos_n": ("m", "position north of home"), "pos_u": ("m", "height above home"),
        "vel_e": ("m/s", "velocity east"), "vel_n": ("m/s", "velocity north"), "vel_u": ("m/s", "climb rate"),
        "acc_e": ("m/s^2", "acceleration east"), "acc_n": ("m/s^2", "acceleration north"), "acc_u": ("m/s^2", "acceleration up"),
        "pos_std_h": ("m", "horizontal position 1-sigma"), "pos_std_v": ("m", "vertical position 1-sigma"),
        "vel_std_h": ("m/s", "horizontal velocity 1-sigma"), "vel_std_v": ("m/s", "vertical velocity 1-sigma"),
        "pos_source": ("enum", "position source"), "reset_counter": ("count", "estimator resets"), "status": ("enum", "estimator health"),
    },
    "arjuna_estimator_innov": {
        "innovation": ("measurement units", "measurement minus prediction"), "innovation_var": ("units^2", "expected spread"),
        "test_ratio": (None, "innovation size vs expected; > 1 = rejected"), "rejected": (None, "rejected this step"),
        "reject_count": ("count", "rejections so far"), "source": ("enum", "measurement source"),
    },
    "arjuna_setpoints": {
        "roll_sp": ("rad", "desired roll"), "pitch_sp": ("rad", "desired pitch"), "yaw_sp": ("rad", "desired yaw"),
        "yaw_rate_sp": ("rad/s", "desired yaw rate"), "thrust_sp": ("0-1", "desired collective thrust"),
        "pos_sp": ("m", "desired position E,N,U"), "vel_sp": ("m/s", "desired velocity E,N,U"), "alt_sp": ("m", "desired altitude"),
    },
    "arjuna_rc_input": {
        "channels": ("us", "raw RC channel"), "rssi": ("%", "receiver signal"), "failsafe": (None, "receiver signal loss"),
        "frame_drops": ("count", "lost RC frames"),
    },
    "arjuna_controller": {
        "setpoint": ("loop units", "desired value"), "measured": ("loop units", "value fed back"),
        "error": ("loop units", "setpoint minus measured"), "p_term": ("output units", "P"), "i_term": ("output units", "I"),
        "d_term": ("output units", "D"), "ff_term": ("output units", "feed-forward"), "output": ("output units", "output"),
        "output_limited": ("output units", "output after limits"), "integrator": ("output units", "integrator state"),
        "saturated": (None, "output hit its limit"), "anti_windup_active": (None, "integrator frozen"),
        "exec_time_us": ("us", "step time"), "deadline_missed": (None, "step finished late"),
    },
    "arjuna_actuator_outputs": {
        "motor_cmd": ("0-1", "mixer output per motor"), "pwm_us": ("us", "ESC signal"), "torque_sp": ("normalised", "torque demand r,p,y"),
        "thrust_sp": ("0-1", "collective thrust demand"), "mixer_saturated": ("bitflags", "motors at min/max"),
        "output_source": ("enum", "main or fallback"), "motor_count": ("count", "motors in use"),
    },
    "arjuna_esc_status": {
        "rpm": ("rpm", "motor speed"), "current": ("A", "motor current"), "voltage": ("V", "ESC input voltage"),
        "temperature": ("degC", "ESC temperature"), "error_count": ("count", "ESC errors"), "fault": ("bitflags", "fault per motor"),
        "online": ("bitflags", "ESC responding"),
    },
    "arjuna_battery": {
        "voltage": ("V", "pack voltage"), "current": ("A", "pack current"), "power": ("W", "power"),
        "consumed_mah": ("mAh", "charge used"), "consumed_wh": ("Wh", "energy used"), "remaining": ("%", "remaining"),
        "temperature": ("degC", "pack temperature"), "cell_count": ("count", "cells in series"), "cell_voltage": ("V", "cell voltage"),
        "internal_resistance": ("ohm", "estimated internal resistance"), "rail_5v": ("V", "5 V rail"), "rail_3v3": ("V", "3.3 V rail"),
        "rail_vccint": ("V", "FPGA core voltage"), "status": ("enum", "battery status"),
    },
    "arjuna_fpga_timing": {
        "period_us": ("us", "configured period"), "exec_time_avg": ("us", "mean execution time"), "exec_time_max": ("us", "max execution time"),
        "jitter_max": ("us", "largest deviation from period"), "data_age_max": ("us", "oldest input used"),
        "executions": ("count", "runs in interval"), "deadline_misses": ("count", "late runs"), "overruns": ("count", "overlapping runs"),
        "fifo_overflows": ("count", "FIFO overflows"), "fifo_underflows": ("count", "FIFO underflows"), "data_valid": (None, "inputs valid"),
    },
    "arjuna_system_health": {
        "cpu_load": ("%", "load per ARM core"), "mem_used": ("KB", "RAM in use"), "fpga_temp": ("degC", "chip temperature"),
        "bus_errors": ("count", "errors on I2C,SPI,UART,CAN,PL-PS"), "watchdog_status": ("enum", "watchdog"),
        "reset_reason": ("enum", "last reset reason"), "log_buffer_fill": ("%", "logger buffer use"),
        "sd_write_max_ms": ("ms", "slowest SD write"), "log_dropouts": ("count", "dropouts so far"),
    },
    "arjuna_communication": {
        "rx_packets": ("count", "received"), "tx_packets": ("count", "sent"), "lost_packets": ("count", "detected losses"),
        "latency": ("us", "delay"), "link_quality": ("%", "link quality"), "rssi": ("dBm", "signal strength"),
        "gcs_connected": (None, "ground station connected"), "heartbeat_age": ("us", "time since heartbeat"), "status": ("enum", "link health"),
    },
    "arjuna_firewall": {
        "cmd_received": ("count", "commands received"), "cmd_accepted": ("count", "accepted"), "cmd_rejected": ("count", "rejected"),
        "last_reject_reason": ("enum", "reject reason"), "signing_active": (None, "message signing enforced"),
    },
    "arjuna_fallback": {
        "state": ("enum", "off/shadow/active"), "switch_reason": ("enum", "switch reason"),
        "primary_heartbeat_age": ("us", "main controller heartbeat age"), "output_diff": ("0-1", "backup vs main motor command"),
        "backup_imu_status": ("enum", "backup IMU health"),
    },
    "arjuna_mission_status": {
        "mission_state": ("enum", "mission state"), "wp_current": ("index", "current waypoint"), "wp_total": ("index", "waypoints"),
        "dist_to_wp": ("m", "distance to waypoint"), "geofence_breach": (None, "outside allowed area"),
    },
    "arjuna_proximity": {"min_distance": ("m", "closest obstacle"), "min_bearing": ("rad", "obstacle bearing"),
                          "avoidance_active": (None, "avoidance active"), "brake_active": (None, "safety brake")},
    "arjuna_flight_event": {"event_code": ("enum", "event"), "severity": ("enum", "severity"), "category": ("enum", "category")},
}

# Angles: the controller's attitude loop (controller_id 1) error/setpoint/measured are radians too.
ANGLE_UNITS = {"rad": "deg", "rad/s": "deg/s"}


def prefix(field: str) -> str:
    return field.split("[", 1)[0]


def unit_of(topic: str, field: str, controller_id: Optional[int] = None) -> Optional[str]:
    if topic == "arjuna_controller" and prefix(field) in ("setpoint", "measured", "error"):
        return {0: "rad/s", 1: "rad"}.get(controller_id) if controller_id is not None else None
    return FIELDS.get(topic, {}).get(prefix(field), (None, ""))[0]


def meaning_of(topic: str, field: str) -> str:
    return FIELDS.get(topic, {}).get(prefix(field), (None, ""))[1]


def display(unit: Optional[str]) -> Tuple[Optional[str], float]:
    """(display unit, scale) — radians become degrees; everything else is unchanged."""
    if unit in ANGLE_UNITS:
        return ANGLE_UNITS[unit], 180.0 / math.pi
    return unit, 1.0
