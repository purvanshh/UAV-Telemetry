"""Pre-flight check: is the drone ready for a mission? Built from a short test flight or a bench run.

A list of plain go / no-go checks, each ``pass``, ``check``, ``fail`` or ``unknown`` (the log does not record what is
needed). A check that cannot be made is never counted as a pass. Limits are the approved triage thresholds, plus one
provisional limit (battery at rest) that is labelled as such. The verdict never says the drone is "safe": a short test
cannot promise that.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from src.arjuna.checks import TH
from src.arjuna.ulog import ArjunaFlight, enum_name, load_json

PF = load_json("preflight.json")
RANK = {"pass": 0, "unknown": 1, "check": 2, "fail": 3}


def _check(cid: str, label: str, status: str, detail: str, action: str = "", provisional: bool = False) -> Dict[str, Any]:
    return {"id": cid, "label": label, "status": status, "detail": detail, "action": action, "provisional": provisional}


def _motors_on(fl: ArjunaFlight) -> Optional[np.ndarray]:
    """Samples of the motor topic while the motors were running (falls back to the armed time, then the whole log)."""
    ao = fl.get("arjuna_actuator_outputs")
    if ao is None:
        return None
    cmds = [ao[f"motor_cmd[{i}]"].to_numpy(float) for i in range(8) if f"motor_cmd[{i}]" in ao]
    cmds = [c for c in cmds if np.isfinite(c).any()]
    if not cmds:
        return None
    return np.nanmean(np.vstack(cmds), axis=0) > PF["motors_running_cmd"]


def _running_window(fl: ArjunaFlight, t: np.ndarray) -> np.ndarray:
    ao = fl.get("arjuna_actuator_outputs")
    on = _motors_on(fl)
    if ao is not None and on is not None and on.any():
        ta = ao["t"].to_numpy(float)
        lo, hi = ta[on].min(), ta[on].max()
        return (t >= lo) & (t <= hi)
    vs = fl.get("arjuna_vehicle_status")
    if vs is not None and "armed" in vs and (vs["armed"].to_numpy(float) > 0).any():
        tv, ar = vs["t"].to_numpy(float), vs["armed"].to_numpy(float) > 0
        return (t >= tv[ar].min()) & (t <= tv[ar].max())
    return np.ones_like(t, dtype=bool)


def _sensors(fl: ArjunaFlight) -> Dict[str, Any]:
    imu = fl.get("arjuna_imu")
    if imu is None:
        return _check("sensors", "Motion sensors", "unknown", "The motion sensor was not recorded.")
    bad = "status" in imu and (imu["status"].to_numpy(float) >= 2).any()
    cols = [c for c in ("gyro[0]", "gyro[1]", "gyro[2]") if c in imu]
    frozen = bool(cols) and all(np.nanstd(imu[c].to_numpy(float)) == 0 for c in cols)
    if bad or frozen:
        return _check("sensors", "Motion sensors", "fail", "The motion sensor reported a fault or sent values that never changed.",
                      "Check the flight controller's connection and recalibrate before flying.")
    return _check("sensors", "Motion sensors", "pass", "The motion sensor was sending changing values.")


def _vibration(fl: ArjunaFlight) -> Dict[str, Any]:
    vb = fl.get("arjuna_vibration")
    if vb is None:
        return _check("vibration", "Shaking (vibration)", "unknown", "Vibration was not recorded.")
    run = _running_window(fl, vb["t"].to_numpy(float))
    axes = [vb[f"accel_vibe[{i}]"].to_numpy(float)[run] for i in range(3) if f"accel_vibe[{i}]" in vb]
    axes = [x[np.isfinite(x)] for x in axes if np.isfinite(x).any()]
    if not axes or not len(axes[0]):
        return _check("vibration", "Shaking (vibration)", "unknown", "Not enough data while the motors were running.")
    p95 = max(np.percentile(x, 95) for x in axes)
    peak = max(x.max() for x in axes)
    clip = 0.0
    if "clip_rate" in vb:
        c = vb["clip_rate"].to_numpy(float)[run]
        clip = float(np.nanmax(c)) if np.isfinite(c).any() else 0.0
    act = "Check every propeller for chips or bends, and make sure the motors and the flight controller are tight."
    if peak > TH["vibe_crit"] or p95 > TH["vibe_warn"] and clip > TH["clip_rate_warn"]:
        return _check("vibration", "Shaking (vibration)", "fail", f"The drone shook hard (up to {peak:.0f} m/s²; the limit is {TH['vibe_warn']:.0f}).", act)
    if p95 > TH["vibe_warn"] or clip > TH["clip_rate_warn"]:
        return _check("vibration", "Shaking (vibration)", "check", f"More shaking than it should have (typical level {p95:.0f} m/s², limit {TH['vibe_warn']:.0f}).", act)
    return _check("vibration", "Shaking (vibration)", "pass", f"Low shaking (typical level {p95:.0f} m/s², limit {TH['vibe_warn']:.0f}).")


def _motors(a: Dict[str, Any]) -> Dict[str, Any]:
    hb = (a.get("physics") or {}).get("hover_balance") or {}
    if hb.get("status") != "computed":
        return _check("motors", "Motors balanced", "unknown",
                      "This needs a few seconds of steady hover. A bench run cannot show it.")
    dev = abs(hb.get("worst_deviation_pct") or 0)
    lim = TH["motor_spread_warn"] * 100
    if dev > lim or hb.get("flag"):
        return _check("motors", "Motors balanced", "check",
                      f"{hb.get('worst_motor')} worked {dev:.0f}% {'harder' if hb['worst_deviation_pct'] > 0 else 'less'} than the others in hover (limit {lim:.0f}%).",
                      "Inspect that motor and its propeller, and check the weight balance.")
    return _check("motors", "Motors balanced", "pass", f"All motors shared the work evenly in hover (worst {dev:.0f}%, limit {lim:.0f}%).")


def _battery(fl: ArjunaFlight) -> Dict[str, Any]:
    bt = fl.get("arjuna_battery")
    if bt is None or not np.isfinite(bt["voltage"].to_numpy(float)).any() or np.nanmax(bt["voltage"]) <= 5:
        return _check("battery", "Battery", "unknown", "The battery was not recorded, so it could not be checked.",
                      "Check the battery charge by hand.")
    t, v = bt["t"].to_numpy(float), bt["voltage"].to_numpy(float)
    rest = v[(t <= t[0] + PF["battery_rest_window_s"]) & np.isfinite(v)]
    cc = bt["cell_count"].dropna() if "cell_count" in bt else []
    cells = int(cc.iloc[0]) if len(cc) and 0 < cc.iloc[0] < 15 else max(1, int(round(float(np.nanmax(v)) / 4.0)))
    per = float(np.median(rest)) / cells
    ok = PF["battery_rest_cell_v_ok"]["value"]
    if per < TH["cell_min_warn"]:
        return _check("battery", "Battery", "fail", f"The battery was very low at the start ({per:.2f} V per cell).", "Charge or replace the battery before flying.")
    if per < ok:
        return _check("battery", "Battery", "check", f"The battery was lower than a full one at the start ({per:.2f} V per cell; a healthy one reads {ok} V or more).",
                      "Charge the battery fully and check it for a weak cell.", provisional=True)
    r5 = bt["rail_5v"].to_numpy(float) if "rail_5v" in bt else None
    if r5 is not None and np.isfinite(r5).any() and np.nanmedian(r5) - np.nanmin(r5) > TH["rail5v_dip_warn"]:
        return _check("battery", "Battery", "check", "The electronics' power supply dipped during the test.", "Check the power connectors and wiring.")
    return _check("battery", "Battery", "pass", f"The battery started healthy ({per:.2f} V per cell).", provisional=True)


def _gps(fl: ArjunaFlight) -> Dict[str, Any]:
    g = fl.get("arjuna_gps")
    if g is None:
        return _check("gps", "GPS", "unknown", "No GPS was recorded (none fitted, or not logged).")
    fix, sats, hacc = (g[c].to_numpy(float) for c in ("fix_type", "satellites", "h_acc"))
    end = slice(max(0, len(g) - max(5, len(g) // 5)), None)
    f, s, h = np.nanmin(fix[end]), np.nanmin(sats[end]), np.nanmax(hacc[end])
    act = "Test in an open area away from buildings and metal, and check the GPS antenna and cable."
    if f < 2:
        return _check("gps", "GPS", "fail", "The GPS did not have a good position fix at the end of the test.", act)
    if s < TH["gps_sats_warn"] or h > TH["gps_hacc_warn"]:
        return _check("gps", "GPS", "check", f"The GPS was weak ({s:.0f} satellites, position error up to {h:.1f} m).", act)
    return _check("gps", "GPS", "pass", f"The GPS was strong ({s:.0f} satellites, position error {h:.1f} m or better).")


def _link(fl: ArjunaFlight) -> Dict[str, Any]:
    rc = fl.get("arjuna_rc_input")
    if rc is None or "failsafe" not in rc or not np.isfinite(rc["failsafe"].to_numpy(float)).any():
        return _check("link", "Radio link", "unknown", "The radio link was not recorded.")
    if (rc["failsafe"].to_numpy(float) > 0).any():
        return _check("link", "Radio link", "fail", "The remote-control signal was lost during the test.", "Check the antennas and range-test the radio link.")
    return _check("link", "Radio link", "pass", "The remote-control signal stayed connected.")


def _warnings(fl: ArjunaFlight) -> Dict[str, Any]:
    vs = fl.get("arjuna_vehicle_status")
    reasons = []
    if vs is not None and "failsafe_active" in vs and (vs["failsafe_active"].to_numpy(float) > 0).any():
        why = vs.loc[vs["failsafe_active"] > 0, "failsafe_reason"].dropna().unique() if "failsafe_reason" in vs else []
        names = ", ".join(enum_name("FAILSAFE_REASON", w).replace("_", " ") for w in why if w) or "unknown reason"
        reasons.append(f"a safety mode switched on ({names})")
    if vs is not None and "prearm_failures" in vs and (np.nan_to_num(vs["prearm_failures"].to_numpy(float)) > 0).any():
        reasons.append("a start-up check failed")
    fe = fl.get("arjuna_flight_event")
    if fe is not None and "severity" in fe and (fe["severity"].to_numpy(float) >= 2).any():
        reasons.append("the flight controller reported a serious event")
    if reasons:
        return _check("warnings", "Flight controller warnings", "fail", "During the test, " + " and ".join(reasons) + ".",
                      "Find out why the flight controller raised the warning, and fix it, before flying again.")
    return _check("warnings", "Flight controller warnings", "pass", "The flight controller raised no safety warnings.")


def _recording(fl: ArjunaFlight, a: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    integ = (a.get("integrity") or {}).get("result")
    if integ == "modified":
        out.append(_check("integrity", "Recording unchanged", "fail", "The recording was changed after the test, so it cannot be trusted.",
                          "Do not rely on this check; repeat the test."))
    if fl.dropouts:
        out.append(_check("gaps", "Complete recording", "check", f"The recording has {len(fl.dropouts)} gap(s), so parts of the test could not be checked.",
                          "Replace the SD card if it is old or slow."))
    return out


def preflight_check(fl: ArjunaFlight, a: Dict[str, Any]) -> Dict[str, Any]:
    if a.get("refused"):
        return {"status": "unreadable", "status_text": "Cannot tell", "status_detail": "This log could not be read, so nothing was checked.",
                "checks": [], "actions": []}
    checks = [_sensors(fl), _vibration(fl), _motors(a), _battery(fl), _gps(fl), _link(fl), _warnings(fl), *_recording(fl, a)]
    # problems the flight checks found in a real test flight
    if (a.get("classification") or {}).get("kind") == "flight":
        crit = [f for f in a.get("findings") or [] if f["severity"] == "critical"]
        warn = [f for f in a.get("findings") or [] if f["severity"] == "warning"]
        if crit:
            checks.append(_check("flight", "The test flight", "fail", f"A serious problem was found during the test ({crit[0]['title']}).",
                                 "Fix the serious problems listed under Problems found, then repeat the test."))
        elif warn:
            checks.append(_check("flight", "The test flight", "check", f"{len(warn)} thing(s) worth checking came up during the test.",
                                 "See the problems below."))
        else:
            checks.append(_check("flight", "The test flight", "pass", "The test flight showed no problems."))
    else:
        checks.append(_check("flight", "The test flight", "unknown",
                             "The drone did not fly (a bench or short ground run), so flying behaviour was not tested."))
    n = {s: sum(c["status"] == s for c in checks) for s in RANK}
    core = {c["id"]: c["status"] for c in checks}
    if n["fail"]:
        status, text = "not_ready", "Not ready for a mission"
        detail = f"{n['fail']} {'check' if n['fail'] == 1 else 'checks'} failed. Fix {'it' if n['fail'] == 1 else 'them'} and repeat the test."
    elif n["check"]:
        status, text = "check", "Check before the mission"
        detail = f"{n['check']} {'check needs' if n['check'] == 1 else 'checks need'} a look before you fly."
    elif core.get("vibration") == "unknown" and core.get("battery") == "unknown":
        status, text = "partial", "Cannot tell"
        detail = "The most important checks (shaking and battery) were not recorded."
    elif n["unknown"]:
        status, text = "ready", "Ready for a mission"
        names = ", ".join(c["label"].lower() for c in checks if c["status"] == "unknown")
        detail = f"Everything that could be checked passed. Not checked, because this log does not record it: {names}."
    else:
        status, text = "ready", "Ready for a mission"
        detail = "Everything passed."
    actions = []
    for c in sorted(checks, key=lambda c: -RANK[c["status"]]):
        if c["status"] in ("fail", "check") and c["action"] and c["action"] not in actions:
            actions.append(c["action"])
    return {"status": status, "status_text": text, "status_detail": detail, "checks": checks, "actions": actions[:5],
            "caveat": "A short test cannot promise that the drone is safe. It shows what this test found.",
            "provisional_limits": [c["id"] for c in checks if c["provisional"]]}
