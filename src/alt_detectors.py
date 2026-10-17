"""Alternative window scorers evaluated against the aggregate LSTM baseline.

All detectors score the same kind of object — a window of the 10 Hz aligned frame
— and flag when ``score > τ``. The deployed LSTM and its threshold are untouched;
``lstm_aggregate`` *is* the baseline. Alternatives that cannot be derived from the
deployed model (statistics on the window, cross-signal consistency) are side
detectors. Window length can only vary for those: the LSTM input is fixed at 30.

Everything a detector needs from reference data (feature error quantiles, scale
statistics, per-phase quantiles) is fitted from *burden-set* windows only (see
``fit_reference``); the threshold is then matched to the baseline's alert burden
on that same set (``match_threshold``). Nothing is fitted on evaluation flights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.context_features import PHASES, window_phase
from src.detector import UAVDetector

MIN_PHASE_WINDOWS = 30  # per-phase normalisation needs at least this many reference windows
Q_REF = 0.99


@dataclass
class WindowSet:
    win: int
    stride: int
    meta: pd.DataFrame  # window_index, session_id, t_start, t_end
    Xz: np.ndarray  # (N, W, F) standardised with the deployed scaler
    rate_res: np.ndarray  # (N, W, 3) derived − gyro, rad/s
    cmd: np.ndarray  # (N, W, 3) roll, pitch, yaw tracking error (deg), yaw may be NaN
    vert_abs: np.ndarray  # (N,) |net altitude change| / duration, m/s
    gyro_rms: np.ndarray  # (N,) rad/s
    phase: np.ndarray  # (N,) object
    lstm: Optional[np.ndarray] = None  # (N,) only when win == 30
    lstm_feat: Optional[np.ndarray] = None  # (N, F)

    @property
    def n(self) -> int:
        return len(self.meta)


def window_indices(aligned: pd.DataFrame, features: Sequence[str], win: int, stride: int) -> np.ndarray:
    """Index arrays of valid windows, in the same order/validity as ``make_windows``."""
    sid = aligned["_session_id"].to_numpy()
    fin = np.isfinite(aligned[list(features)].to_numpy(dtype=np.float64)).all(axis=1)
    out = []
    for s in np.unique(sid):
        if s < 0:
            continue
        idx = np.flatnonzero(sid == s)
        if len(idx) < win:
            continue
        for s0 in range(0, len(idx) - win + 1, stride):
            w = idx[s0 : s0 + win]
            if fin[w].all():
                out.append(w)
    return np.stack(out) if out else np.empty((0, win), dtype=int)


def build_windowset(
    aligned: pd.DataFrame,
    sig: pd.DataFrame,
    features: Sequence[str],
    mu: np.ndarray,
    sd: np.ndarray,
    win: int,
    stride: int,
    detector: Optional[UAVDetector] = None,
) -> WindowSet:
    W = window_indices(aligned, features, win, stride)
    t = aligned["_t_s"].to_numpy()
    sid = aligned["_session_id"].to_numpy()
    X = aligned[list(features)].to_numpy(dtype=np.float32)
    n = len(W)
    meta = pd.DataFrame({
        "window_index": np.arange(n),
        "session_id": sid[W[:, 0]] if n else np.array([], dtype=int),
        "t_start": t[W[:, 0]] if n else np.array([]),
        "t_end": t[W[:, -1]] if n else np.array([]),
    })
    Xz = ((X[W] - mu) / sd).astype(np.float32) if n else np.empty((0, win, len(features)), np.float32)
    rr = sig[["rate_res_x", "rate_res_y", "rate_res_z"]].to_numpy(dtype=float)
    cm = sig[["cmd_roll", "cmd_pitch", "cmd_yaw"]].to_numpy(dtype=float)
    alt = sig["alt"].to_numpy(dtype=float)
    gn = sig["gyro_norm"].to_numpy(dtype=float)
    dur = max((win - 1) * float(np.median(np.diff(t))), 1e-9) if len(t) > 1 else 1.0
    vert = np.abs(alt[W[:, -1]] - alt[W[:, 0]]) / dur if n else np.array([])  # net rate, m/s
    grms = np.sqrt(np.nanmean(gn[W] ** 2, axis=1)) if n else np.array([])
    ws = WindowSet(
        win=win, stride=stride, meta=meta, Xz=Xz,
        rate_res=rr[W] if n else np.empty((0, win, 3)),
        cmd=cm[W] if n else np.empty((0, win, 3)),
        vert_abs=vert, gyro_rms=grms, phase=window_phase(vert, grms) if n else np.array([], dtype=object),
    )
    if win == 30 and detector is not None and n:
        raw = X[W]
        ws.lstm = detector.recon_error_per_window(raw).astype(np.float64)
        ws.lstm_feat = detector.recon_error_per_feature(raw).astype(np.float64)
    return ws


# ---------------------------------------------------------------- components
def _nanrms(a: np.ndarray, axes) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.sqrt(np.nanmean(a**2, axis=axes))


def components(ws: WindowSet, ref: Optional["Reference"] = None) -> Dict[str, np.ndarray]:
    c: Dict[str, np.ndarray] = {}
    if ws.n == 0:
        return c
    if ws.lstm is not None:
        c["lstm"] = ws.lstm
        if ref is not None and ref.feat_q is not None:
            c["lstm_pf"] = (ws.lstm_feat / ref.feat_q).max(axis=1)
    c["zmax"] = np.abs(ws.Xz).max(axis=(1, 2)).astype(np.float64)
    d = np.diff(ws.Xz, axis=1)
    c["diffe"] = (d**2).mean(axis=(1, 2)).astype(np.float64)
    if ref is not None and ws.win in ref.flat_med:
        med = ref.flat_med[ws.win]
        sd = ws.Xz.std(axis=1)
        c["flat"] = (med / np.maximum(sd, 1e-3 * med)).max(axis=1).astype(np.float64)
    c["rate"] = _nanrms(ws.rate_res, (1, 2))
    c["cmd"] = _nanrms(ws.cmd, (1, 2))
    return c


@dataclass
class Reference:
    """Statistics fitted on burden-set windows only."""

    feat_q: Optional[np.ndarray] = None  # (F,) per-feature error quantile
    flat_med: Dict[int, np.ndarray] = field(default_factory=dict)  # win -> (F,) median window std
    comp_q: Dict[str, float] = field(default_factory=dict)  # component -> global Q99
    phase_q: Dict[str, Dict[str, Optional[float]]] = field(default_factory=dict)  # comp -> phase -> Q99|None
    n_windows: Dict[int, int] = field(default_factory=dict)


def fit_reference(sets: Mapping[int, Sequence[WindowSet]]) -> Reference:
    """``sets[win]`` = burden-set window sets (one per flight) for that window length."""
    ref = Reference()
    w30 = [s for s in sets.get(30, []) if s.n]
    if w30:
        fe = np.concatenate([s.lstm_feat for s in w30])
        ref.feat_q = np.quantile(fe, Q_REF, axis=0)
    for win, lst in sets.items():
        lst = [s for s in lst if s.n]
        ref.n_windows[win] = int(sum(s.n for s in lst))
        if lst:
            ref.flat_med[win] = np.median(np.concatenate([s.Xz.std(axis=1) for s in lst]), axis=0)
    if w30:
        # components for normalising fused detectors (global + per phase)
        comps = [components(s, ref) for s in w30]
        phases = np.concatenate([s.phase for s in w30])
        for k in ("lstm", "rate", "zmax", "diffe", "flat", "cmd"):
            v = np.concatenate([c[k] for c in comps])
            fin = np.isfinite(v)
            ref.comp_q[k] = float(np.quantile(v[fin], Q_REF)) if fin.any() else np.nan
            ref.phase_q[k] = {}
            for p in PHASES:
                m = fin & (phases == p)
                ref.phase_q[k][p] = float(np.quantile(v[m], Q_REF)) if m.sum() >= MIN_PHASE_WINDOWS else None
    return ref


def _phase_norm(vals: np.ndarray, ws: WindowSet, ref: Reference, comp: str) -> np.ndarray:
    out = np.empty(len(vals))
    for p in PHASES:
        q = ref.phase_q[comp].get(p) or ref.comp_q[comp]  # fallback: global when phase under-sampled
        m = ws.phase == p
        out[m] = vals[m] / q
    return out


# ---------------------------------------------------------------- detectors
@dataclass(frozen=True)
class DetectorDef:
    name: str
    family: str  # baseline | phase3 | phase4
    win: int
    description: str
    score: Callable[[WindowSet, Reference], np.ndarray]
    needs_rate_convention: bool = False


def _comp(key: str):
    return lambda ws, ref: components(ws, ref)[key]


def _fused(keys: Sequence[str]):
    def f(ws, ref):
        c = components(ws, ref)
        arr = np.stack([c[k] / ref.comp_q[k] for k in keys])
        return np.where(np.isnan(arr).any(axis=0), np.nan, arr.max(axis=0))
    return f


def _fused_phase(keys: Sequence[str]):
    def f(ws, ref):
        c = components(ws, ref)
        arr = np.stack([_phase_norm(c[k], ws, ref, k) for k in keys])
        return np.where(np.isnan(arr).any(axis=0), np.nan, arr.max(axis=0))
    return f


DETECTORS: List[DetectorDef] = [
    DetectorDef("lstm_aggregate", "baseline", 30, "Deployed LSTM-AE aggregate reconstruction MSE (baseline).", _comp("lstm")),
    DetectorDef("lstm_per_feature", "phase3", 30, "Max over features of per-feature LSTM error ÷ that feature's reference Q99 (per-feature thresholds).", _comp("lstm_pf")),
    DetectorDef("zmax_w30", "phase3", 30, "Max |standardised value| in the window.", _comp("zmax")),
    DetectorDef("zmax_w10", "phase3", 10, "Max |standardised value|, 1 s windows.", _comp("zmax")),
    DetectorDef("diffe_w30", "phase3", 30, "Mean squared first difference of standardised features (spikes/oscillation).", _comp("diffe")),
    DetectorDef("diffe_w10", "phase3", 10, "Mean squared first difference, 1 s windows.", _comp("diffe")),
    DetectorDef("flat_w30", "phase3", 30, "Flatline: max over features of reference window std ÷ window std (stuck values).", _comp("flat")),
    DetectorDef("flat_w10", "phase3", 10, "Flatline, 1 s windows.", _comp("flat")),
    DetectorDef("rate_consistency_w30", "phase4", 30, "RMS of (Euler-derived body rate − gyro): attitude/gyro consistency.", _comp("rate"), True),
    DetectorDef("cmd_tracking_w30", "phase4", 30, "RMS of commanded−actual roll/pitch/yaw.", _comp("cmd")),
    DetectorDef("fused_lstm_rate", "phase4", 30, "max(LSTM ÷ Q99, rate consistency ÷ Q99).", _fused(("lstm", "rate")), True),
    DetectorDef("fused_lstm_rate_phase", "phase4", 30, "Same, each term normalised by its flight-phase Q99 (global where the phase is under-sampled).", _fused_phase(("lstm", "rate")), True),
    DetectorDef("fused_lstm_cmd", "phase4", 30, "max(LSTM ÷ Q99, command tracking ÷ Q99).", _fused(("lstm", "cmd"))),
    DetectorDef("fused_all", "phase4", 30, "max over LSTM, rate consistency, zmax, diffe, flatline, each ÷ its Q99.", _fused(("lstm", "rate", "zmax", "diffe", "flat")), True),
]


def match_threshold(scores: np.ndarray, target_fraction: float) -> float:
    """τ such that ``score > τ`` flags ≈ ``target_fraction`` of the finite reference windows."""
    s = np.sort(scores[np.isfinite(scores)])[::-1]
    if len(s) == 0:
        return float("nan")
    k = int(round(target_fraction * len(s)))
    return float(s[min(k, len(s) - 1)]) if k > 0 else float(s[0])
