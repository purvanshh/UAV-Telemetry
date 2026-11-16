"""Physics layer: what the quadcopter should have done, compared with what it did (Roadmap, "Three ways to use the
mathematical model"). Airframe constants: ``config/arjuna/airframe.json``.

1. Hover motor balance (static torque balance). In steady hover the net torque is zero, so a motor that needs a
   larger command than the others is weaker, or the centre of gravity / an arm is off. Works at any log rate; this is
   the check the approved triage uses for MOTOR_ESC.

2. Vertical thrust residual. Predicted specific force along body z, k * sum(u_i^2) / m (scaled by (V/V0)^2 when the
   battery is logged), against the IMU's measured specific force. k is calibrated on this flight's steady hover, so
   the residual shows *changes* in thrust per command (thrust loss, payload change), not absolute calibration.

3. Rotational dynamics and per-motor effectiveness (needs body rates and motor outputs at >= 100 Hz)::

       F_i = lag_tau(u_i^2)
       tau = [ d * sum y_i eta_i F_i,  -d * sum x_i eta_i F_i,  c_q * sum s_i eta_i F_i ]
       J w_dot + w x J w = B diag(eta) F            (J = I / thrust scale, diagonal)

   Fitted by least squares with sum(eta) = 4 (the average motor is the reference), first over the whole air time for
   J and eta, then eta per sliding window with J fixed. eta_i = 1 is nominal; 0.8 means motor i makes 20% less
   thrust than commanded relative to the others. The controller hides this in the attitude; it cannot hide it here.

No physics threshold is approved yet, so (2) is reported as observations and (3) as medium-confidence findings,
flagged with the approved 12% motor-spread limit applied to effectiveness.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Tuple

import numpy as np

from src.arjuna.checks import TH, _r, clock, ev, intervals
from src.arjuna.ulog import ArjunaFlight, load_json

AF = load_json("airframe.json")
G = 9.80665
GROUND_EFFECT_MASK_S = 3.0   # ignore this long after take-off and before landing
ETA_WINDOW_S = 10.0


def _motors() -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    ms = sorted(AF["motors"], key=lambda m: m["index"])
    return (np.array([m["x"] for m in ms], float), np.array([m["y"] for m in ms], float),
            np.array([m["spin"] for m in ms], float), [m["name"] for m in ms])


def layout_known(fl: ArjunaFlight) -> bool:
    return str(fl.info.get("motor_layout", "known")) != "unknown"


def motor_names(fl: ArjunaFlight) -> List[str]:
    """Position names from the airframe, or plain m1..mN when the source does not say which corner is which."""
    names = _motors()[3]
    return names if layout_known(fl) else [f"m{i + 1}" for i in range(len(names))]


def _interp(t_src: np.ndarray, y: np.ndarray, t: np.ndarray) -> np.ndarray:
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return np.full_like(t, np.nan, dtype=float)
    return np.interp(t, t_src[ok], y[ok], left=np.nan, right=np.nan)


def _air_core(fl: ArjunaFlight, t: np.ndarray) -> np.ndarray:
    """In the air, excluding the ground-effect margins after take-off and before landing."""
    m = np.zeros_like(t, dtype=bool)
    for a, b in fl.air_intervals():
        m |= (t >= a + GROUND_EFFECT_MASK_S) & (t <= b - GROUND_EFFECT_MASK_S)
    return m


# ---- 1. hover balance -----------------------------------------------------------------------------------------------

def hover_balance(fl: ArjunaFlight) -> Dict[str, Any]:
    ao, at = fl.get("arjuna_actuator_outputs"), fl.get("arjuna_attitude")
    if ao is None or at is None:
        return {"status": "not_computable", "reason": "needs arjuna_actuator_outputs and arjuna_attitude"}
    x, y, s, _ = _motors()
    names = motor_names(fl)
    n = len(names)
    if not all(f"motor_cmd[{i}]" in ao for i in range(n)):
        return {"status": "not_computable", "reason": f"needs motor_cmd for {n} motors"}
    ta = ao["t"].to_numpy(float)
    m = fl.in_air(ta)
    cmd = np.vstack([ao[f"motor_cmd[{i}]"].to_numpy(float) for i in range(n)])
    rates = np.vstack([_interp(at["t"].to_numpy(float), at[f"rates[{i}]"].to_numpy(float), ta) for i in range(3)])
    roll = _interp(at["t"].to_numpy(float), at["roll"].to_numpy(float), ta)
    pitch = _interp(at["t"].to_numpy(float), at["pitch"].to_numpy(float), ta)
    steady = m & (np.nanmax(np.abs(rates), axis=0) < 0.15) & (np.abs(roll) < 0.1) & (np.abs(pitch) < 0.1) & (np.nanmin(cmd, axis=0) > 0.05)
    dt = float(np.median(np.diff(ta))) if len(ta) > 1 else 0.1
    hover_s = float(steady.sum() * dt)
    out: Dict[str, Any] = {"status": "computed", "steady_hover_s": _r(hover_s, 1), "motors": names,
                           "criteria": "in the air, |body rates| < 0.15 rad/s, |roll|,|pitch| < 0.1 rad, every motor > 0.05"}
    if hover_s <= 5:
        return {**out, "status": "not_computable", "reason": f"only {hover_s:.1f} s of steady hover (needs more than 5 s)"}
    mc = np.nanmean(cmd[:, steady], axis=1)
    dev = mc / np.mean(mc) - 1
    eff = (np.mean(mc) / mc) ** 2  # with F proportional to u^2: relative effectiveness that keeps torque balanced
    proj = {"roll_left_heavy_pct": float(np.dot(dev, y) / n * 100), "pitch_nose_heavy_pct": float(np.dot(dev, x) / n * 100),
            "yaw_cw_harder_pct": float(np.dot(dev, s) / n * 100)}
    mags = np.abs(list(proj.values()))
    single = bool(mags.max() > 0 and mags.min() / mags.max() > 0.5)
    worst = int(np.argmax(np.abs(dev)))
    out.update({
        "mean_cmd": [_r(v, 4) for v in mc], "deviation_pct": [_r(v * 100, 1) for v in dev],
        "relative_effectiveness": [_r(v, 3) for v in eff], "imbalance": {k: _r(v, 1) for k, v in proj.items()} if layout_known(fl) else None,
        "worst_motor": names[worst], "worst_deviation_pct": _r(dev[worst] * 100, 1),
        "pattern": "motor positions unknown" if not layout_known(fl) else ("one motor differs from the other three (prop, motor or ESC)" if single else
                    "a roll/pitch pattern (centre of gravity offset or payload)" if max(mags[:2]) >= mags[2] else
                    "a yaw pattern (twisted arm, mismatched props or motor timing)"),
        "interpretation": (_interpret(names, dev, worst, single, proj) if layout_known(fl) else
                           f"{names[worst]} needs {abs(dev[worst]) * 100:.0f}% {'more' if dev[worst] > 0 else 'less'} command than the "
                           f"motor average to hold hover. Which corner each motor sits on is not known for this source, so no "
                           f"centre-of-gravity, arm or diagonal conclusion is drawn."),
        "flag": bool(abs(dev[worst]) > TH["motor_spread_warn"]),
        "t_range": [_r(ta[steady][0], 1), _r(ta[steady][-1], 1)],
    })
    return out


def _interpret(names: List[str], dev: np.ndarray, worst: int, single: bool, proj: Dict[str, float]) -> str:
    """Plain reading of the hover balance, written by code so the direction is never inverted."""
    x, y, s, _ = _motors()
    nm, pct = names[worst], abs(dev[worst]) * 100
    opp = int(np.argmin(x * x[worst] + y * y[worst]))  # the diagonally opposite motor
    if dev[worst] > 0:
        head = (f"{nm} needs {pct:.0f}% more command than the motor average to hold hover. That means {nm} (its propeller, "
                f"motor or ESC) gives less thrust per command than the others, or the centre of gravity sits toward {nm}.")
    else:
        head = (f"{nm} needs {pct:.0f}% less command than the motor average to hold hover. That means {nm} gives more "
                f"thrust per command than the others, or the other motors (most likely the diagonal opposite, "
                f"{names[opp]}) give less, or the centre of gravity sits away from {nm}.")
    if single:
        tail = " The pattern is one motor against the other three, which points to that motor's propulsion path rather than the load."
    elif max(abs(proj["roll_left_heavy_pct"]), abs(proj["pitch_nose_heavy_pct"])) >= abs(proj["yaw_cw_harder_pct"]):
        side = ("left" if proj["roll_left_heavy_pct"] > 0 else "right") if abs(proj["roll_left_heavy_pct"]) >= abs(proj["pitch_nose_heavy_pct"]) \
            else ("nose" if proj["pitch_nose_heavy_pct"] > 0 else "tail")
        tail = f" The pattern is a pair pattern: the vehicle is heavy on the {side} side (centre of gravity or payload), not one motor."
    else:
        tail = (" The pattern is a yaw pattern: the " + ("clockwise" if proj["yaw_cw_harder_pct"] > 0 else "anticlockwise")
                + " motors work harder, which points to a twisted arm, mismatched propellers or motor timing.")
    return head + tail


# ---- 2. vertical thrust residual ------------------------------------------------------------------------------------

def thrust_residual(fl: ArjunaFlight) -> Dict[str, Any]:
    ao, imu = fl.get("arjuna_actuator_outputs"), fl.get("arjuna_imu")
    if ao is None or imu is None or "accel[2]" not in imu:
        return {"status": "not_computable", "reason": "needs arjuna_actuator_outputs and arjuna_imu accel"}
    n = len(AF["motors"])
    ta = ao["t"].to_numpy(float)
    u = np.vstack([ao[f"motor_cmd[{i}]"].to_numpy(float) for i in range(n) if f"motor_cmd[{i}]" in ao])
    if u.shape[0] != n:
        return {"status": "not_computable", "reason": "motor commands missing"}
    az = _interp(imu["t"].to_numpy(float), imu["accel[2]"].to_numpy(float), ta)
    drive = np.nansum(u ** AF["thrust_exponent"], axis=0)
    volt_note = "battery not logged; thrust per command assumed constant"
    bt = fl.get("arjuna_battery")
    if bt is not None and np.isfinite(bt["voltage"].to_numpy(float)).any() and np.nanmax(bt["voltage"]) > 5:
        v = _interp(bt["t"].to_numpy(float), bt["voltage"].to_numpy(float), ta)
        v0 = float(np.nanmedian(v[fl.in_air(ta)])) if fl.in_air(ta).any() else float(np.nanmedian(v))
        if np.isfinite(v0) and v0 > 0:
            drive = drive * np.where(np.isfinite(v), (v / v0) ** 2, 1.0)
            volt_note = "scaled by (V / median V in air)^2"
    core = _air_core(fl, ta) & np.isfinite(az) & (drive > 0)
    if core.sum() < 50:
        return {"status": "not_computable", "reason": "too little air time outside take-off and landing"}
    k = float(np.nanmedian(az[core] / drive[core]))
    pred = k * drive
    r = np.where(core, az - pred, np.nan)
    med = float(np.nanmedian(r))
    mad = float(np.nanmedian(np.abs(r - med)) * 1.4826) or 1e-6
    dt = float(np.median(np.diff(ta))) if len(ta) > 1 else 0.1
    obs = []
    for sign, label in ((-1, "less upward acceleration than the motor commands predict (thrust shortfall)"),
                        (1, "more upward acceleration than predicted (lighter, or external lift)")):
        bad = (sign * (r - med) > 4 * mad)
        for a, b in intervals(ta, bad, merge_gap=1.0, min_len=1.0):
            sel = (ta >= a) & (ta <= b)
            obs.append({"t_start": _r(a, 2), "t_end": _r(b, 2), "when": f"{clock(a)}–{clock(b)}", "direction": label,
                        "median_residual_mps2": _r(np.nanmedian(r[sel]), 3), "residual_spread_mps2": _r(mad, 3)})
    obs.sort(key=lambda o: -abs(o["median_residual_mps2"] or 0))
    return {"status": "computed", "model": "a_z = k * sum(u_i^2) / m" + ("" if "not" in volt_note else " * (V/V0)^2"),
            "voltage_compensation": volt_note, "k_calibrated": _r(k, 4), "samples": int(core.sum()),
            "residual_median_mps2": _r(med, 3), "residual_robust_spread_mps2": _r(mad, 3),
            "residual_rms_mps2": _r(float(np.sqrt(np.nanmean((r - med) ** 2))), 3),
            "sustained_departures": obs[:10], "rate_hz": _r(1 / dt, 1),
            "threshold_note": "Departures beyond 4 robust spreads for at least 1 s. Not an approved threshold: the healthy-flight "
                              "residual band has not been measured yet (Roadmap phase 3)."}


# ---- 3. rotational dynamics and eta ---------------------------------------------------------------------------------

def _lag(F: np.ndarray, dt: float, tau: float) -> np.ndarray:
    a = dt / (tau + dt)
    out = np.empty_like(F)
    out[:, 0] = F[:, 0]
    for k in range(1, F.shape[1]):
        out[:, k] = out[:, k - 1] + a * (F[:, k] - out[:, k - 1])
    return out


def _design(w: np.ndarray, wdot: np.ndarray, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rows for  B diag(eta) F - (J w_dot + w x J w) = 0  in theta = [eta_1..eta_n, Jx, Jy, Jz]."""
    x, y, s, _ = _motors()
    d, cq = AF["arm_projection_m"], AF["yaw_torque_coeff_cq"]
    B = np.vstack([d * y, -d * x, cq * s])                 # 3 x n
    n, T = F.shape
    rows = []
    wx, wy, wz = w
    # (w x J w) = [(Jz - Jy) wy wz, (Jx - Jz) wz wx, (Jy - Jx) wx wy]
    cross = {0: {2: wy * wz, 1: -wy * wz}, 1: {0: wz * wx, 2: -wz * wx}, 2: {1: wx * wy, 0: -wx * wy}}
    for a in range(3):
        R = np.zeros((T, n + 3))
        R[:, :n] = (B[a][:, None] * F).T
        R[:, n + a] -= wdot[a]
        for j, coef in cross[a].items():
            R[:, n + j] -= coef
        rows.append(R)
    return np.vstack(rows), B


def _constrained_ls(A: np.ndarray, n: int) -> np.ndarray:
    """argmin ||A theta||^2 subject to sum(eta) = n."""
    c = np.zeros(A.shape[1])
    c[:n] = 1.0
    M = A.T @ A + 1e-12 * np.eye(A.shape[1]) * np.trace(A.T @ A) / A.shape[1]
    z = np.linalg.solve(M, c)
    return n * z / float(c @ z)


def dynamics_fit(fl: ArjunaFlight) -> Dict[str, Any]:
    ao, at = fl.get("arjuna_actuator_outputs"), fl.get("arjuna_attitude")
    need = AF["min_rate_hz_for_dynamics"]
    if ao is None or at is None:
        return {"status": "not_computable", "reason": "needs arjuna_actuator_outputs and arjuna_attitude"}
    if not layout_known(fl):
        return {"status": "not_computable", "reason": "motor positions are not known for this source, and the torque model "
                                                      "needs each motor's corner and spin direction"}
    rate_src = fl.get("arjuna_imu_filtered")
    rate_topic = "arjuna_imu_filtered" if rate_src is not None and "gyro_filtered[0]" in rate_src else "arjuna_attitude"
    src = rate_src if rate_topic == "arjuna_imu_filtered" else at
    cols = [f"gyro_filtered[{i}]" for i in range(3)] if rate_topic == "arjuna_imu_filtered" else [f"rates[{i}]" for i in range(3)]
    tr, tm = src["t"].to_numpy(float), ao["t"].to_numpy(float)
    hz = min(_hz(tr), _hz(tm))
    if hz < need:
        return {"status": "not_computable", "rate_hz": _r(hz, 1), "required_hz": need,
                "reason": f"body rates and motor outputs are logged at {hz:.0f} Hz; the dynamics fit needs at least {need} Hz "
                          "(ARJUNA engineering profile). Hover balance is used instead."}
    _, _, _, names = _motors()
    n = len(names)
    dt = 1.0 / hz
    t = np.arange(max(tr[0], tm[0]), min(tr[-1], tm[-1]), dt)
    core = _air_core(fl, t)
    if core.sum() < 5 * hz:
        return {"status": "not_computable", "reason": "less than 5 s of usable air time"}
    w = np.vstack([_interp(tr, src[c].to_numpy(float), t) for c in cols])
    u = np.vstack([_interp(tm, ao[f"motor_cmd[{i}]"].to_numpy(float), t) for i in range(n)])
    k = max(1, int(round(0.01 * hz)))  # ~10 ms smoothing before differentiating
    ws = np.vstack([np.convolve(np.nan_to_num(row), np.ones(2 * k + 1) / (2 * k + 1), mode="same") for row in w])
    wdot = np.gradient(ws, dt, axis=1)
    F = _lag(np.nan_to_num(u) ** AF["thrust_exponent"], dt, AF["motor_time_constant_s"])
    ok = core & np.all(np.isfinite(w), axis=0) & np.all(np.isfinite(u), axis=0)
    A, B = _design(ws[:, ok], wdot[:, ok], F[:, ok])
    theta = _constrained_ls(A, n)
    eta, J = theta[:n], theta[n:]
    res = (A @ theta).reshape(3, -1)
    windows = []
    step = ETA_WINDOW_S / 2
    tt = t[ok]
    for a in np.arange(tt[0], tt[-1] - ETA_WINDOW_S / 2, step):
        sel = (tt >= a) & (tt < a + ETA_WINDOW_S)
        if sel.sum() < 2 * hz:
            continue
        Aw, _ = _design(ws[:, ok][:, sel], wdot[:, ok][:, sel], F[:, ok][:, sel])
        lhs = Aw[:, :n]
        rhs = -(Aw[:, n:] @ J)
        ew, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
        windows.append({"t_start": _r(a, 1), "t_end": _r(a + ETA_WINDOW_S, 1), "eta": [_r(v, 3) for v in ew]})
    out = {"status": "computed", "rate_hz": _r(hz, 1), "rate_source": rate_topic, "samples": int(ok.sum()),
           "eta": [_r(v, 3) for v in eta], "motors": names, "J_over_thrust_scale": [_r(v, 6) for v in J],
           "residual_rms": [_r(float(np.sqrt(np.mean(r ** 2))), 5) for r in res], "windows": windows[:120],
           "method": "least squares on J w_dot + w x J w = B diag(eta) lag(u^2), sum(eta) = n; eta per 10 s window with J fixed",
           "flag_rule": f"a motor's eta more than {TH['motor_spread_warn'] * 100:.0f}% below the mean of the others "
                        "(the approved motor-spread limit applied to effectiveness; not separately approved)"}
    rel = np.array([eta[i] / np.mean(np.delete(eta, i)) for i in range(n)])
    out["relative_to_others"] = [_r(v, 3) for v in rel]
    worst = int(np.argmin(rel))
    out["flag_motor"] = names[worst] if rel[worst] < 1 - TH["motor_spread_warn"] else None
    return out


def _hz(t: np.ndarray) -> float:
    d = np.diff(t)
    d = d[d > 0]
    return float(1.0 / np.median(d)) if len(d) else 0.0


# ---- findings --------------------------------------------------------------------------------------------------------

def run_physics(fl: ArjunaFlight) -> Dict[str, Any]:
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        hb, tr, dyn = hover_balance(fl), thrust_residual(fl), dynamics_fit(fl)
    findings: List[Dict[str, Any]] = []
    airs = fl.air_intervals()
    a0, a1 = (airs[0][0], airs[-1][1]) if airs else (0.0, fl.duration_s)
    if hb.get("flag"):
        i = hb["motors"].index(hb["worst_motor"])
        word = "works harder" if hb["worst_deviation_pct"] > 0 else "works less"
        findings.append({"code": "MOTOR_ESC", "severity": "warning",
                         "title": f"{hb['worst_motor']} {word} in hover ({hb['worst_deviation_pct']:+.0f}%)",
                         "t_start": _r(a0, 2), "t_end": _r(a1, 2), "when": f"{clock(a0)}–{clock(a1)}", "window": "whole_flight",
                         "detector": "physics:hover_balance", "confidence": "high",
                         "basis": f"Static torque balance in {hb['steady_hover_s']} s of steady hover; approved limit "
                                  f"{TH['motor_spread_warn'] * 100:.0f}% from the mean. {hb['interpretation']}",
                         "evidence": [ev("arjuna_actuator_outputs", f"motor_cmd[{i}]", "mean in steady hover", hb["mean_cmd"][i], "0-1"),
                                      ev("arjuna_actuator_outputs", f"motor_cmd[{i}]", "deviation from motor mean", hb["worst_deviation_pct"], "%"),
                                      ev("physics", "hover_balance", "steady hover time", hb["steady_hover_s"], "s")]})
    if dyn.get("flag_motor"):
        i = dyn["motors"].index(dyn["flag_motor"])
        findings.append({"code": "MOTOR_ESC", "severity": "warning",
                         "title": f"{dyn['flag_motor']} makes {(1 - dyn['relative_to_others'][i]) * 100:.0f}% less thrust per command than the others",
                         "t_start": _r(a0, 2), "t_end": _r(a1, 2), "when": f"{clock(a0)}–{clock(a1)}", "window": "whole_flight",
                         "detector": "physics:eta_fit", "confidence": "medium", "basis": dyn["flag_rule"],
                         "evidence": [ev("physics", "eta", f"effectiveness of {dyn['flag_motor']}", dyn["eta"][i], None),
                                      ev("physics", "eta", "relative to the mean of the other motors", dyn["relative_to_others"][i], None),
                                      ev("physics", "eta", "samples used", dyn["samples"], "count")]})
    return {"hover_balance": hb, "thrust_residual": tr, "dynamics": dyn, "findings": findings, "airframe": AF}
