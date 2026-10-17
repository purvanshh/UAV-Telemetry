"""Controlled synthetic perturbations for detector sensitivity checks.

**Synthetic results are not real-world detection accuracy.** They measure whether
the deployed detector responds to a *known, injected* pattern in an otherwise
unlabelled source flight, and nothing else.

Perturbations are applied to the **raw** uploaded frames (before alignment) so we
can measure whether preprocessing preserved the pattern. The source frames are
never modified: every injection works on a copy, and generated cases are written
only to the evaluation output directory.

Injection types
---------------
``spike``          offset of ``amplitude_sigma`` σ on ``n_samples`` consecutive raw rows
``drift``          linear ramp 0 → ``amplitude_sigma`` σ across the interval
``stuck``          column frozen at its value at interval start
``missing_values`` column set to NaN over the interval
``dropout``        rows of the stream removed over the interval (source gap)
``oscillation``    ``amplitude_sigma`` σ · sin(2π f (t − t0)) over the interval
``multi_feature``  simultaneous offset of ``amplitude_sigma`` σ on several columns

σ is the deployed scaler std of the corresponding model feature, so amplitudes are
in the units the detector sees.

Outcome definitions (per injection, ``threshold`` τ):

* **eligible** – at least one *scored* window overlaps the injected interval.
  Windows lost to NaN/gaps are not scored; an ineligible injection cannot be
  detected by construction and is excluded from the eligible detection rate (but
  reported).
* **detected** – at least one overlapping scored window has ``score > τ``. The
  injection placement requires every overlapping baseline window to be unflagged
  at the deployed threshold, so the flag is attributable to the injection. At
  other τ, an injection whose *baseline* max overlapping score already exceeds τ is
  ``confounded`` and excluded from that τ's detection rate.
* **latency_s** – ``min(t_end of flagged overlapping windows) − t_start`` of the
  injection, floored at 0: the earliest moment a window containing the pattern
  has been fully observed.
* **pattern_preserved** – after resampling to the 10 Hz grid the affected model
  features still differ from the un-injected baseline by ≥ 50 % of the nominal
  amplitude (value-changing types); ``stuck``: in-interval standard deviation falls
  to ≤ 10 % of baseline; ``missing_values``/``dropout``: grid samples in the
  interval are non-finite / inside a gap.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.artifact_loader import ModelArtifacts
from src.calibration import score_flight_frames
from src.detector import UAVDetector
from src.timebase import find_timestamp_column

INJECTION_TYPES = (
    "spike",
    "drift",
    "stuck",
    "missing_values",
    "dropout",
    "oscillation",
    "multi_feature",
)
# raw column -> (stream, model features it affects)
RAW_COLUMNS: Dict[str, Tuple[str, List[str]]] = {
    "GyrX": ("IMU", ["gyro_x"]),
    "GyrY": ("IMU", ["gyro_y"]),
    "GyrZ": ("IMU", ["gyro_z"]),
    "AccX": ("IMU", ["acc_x"]),
    "AccY": ("IMU", ["acc_y"]),
    "AccZ": ("IMU", ["acc_z"]),
    "Roll": ("ATT", ["roll", "roll_res"]),
    "Pitch": ("ATT", ["pitch", "pitch_res"]),
}
# σ for raw columns that map 1:1 to a model feature (roll_res has its own σ but roll drives the offset)
_SIGMA_FEATURE = {c: v[1][0] for c, v in RAW_COLUMNS.items()}
PRESERVE_FRACTION = 0.5
STUCK_RESIDUAL_FRACTION = 0.1
VACUOUS_BASELINE_SIGMA = 0.02
OSC_HZ = 2.5


@dataclass
class InjectionSpec:
    injection_id: str
    type: str
    columns: List[str]
    t_start: float
    t_end: float
    amplitude_sigma: float = 0.0
    n_samples: int = 3  # spike only
    osc_hz: float = OSC_HZ
    seed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def case_seed(base_seed: int, *parts: Any) -> int:
    h = hashlib.sha256("|".join(map(str, (base_seed, *parts))).encode()).hexdigest()
    return int(h[:8], 16)


def _t_s(df: pd.DataFrame) -> np.ndarray:
    col = find_timestamp_column(list(df.columns))
    if col is None:
        raise ValueError("no timestamp column")
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype="float64") / 1e6


def apply_injection(
    frames: Mapping[str, pd.DataFrame],
    spec: InjectionSpec,
    sigma: Mapping[str, float],
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """Return ``(new_frames, record)``; ``frames`` is never modified."""
    if spec.type not in INJECTION_TYPES:
        raise ValueError(f"unknown injection type {spec.type}")
    out = {k: v for k, v in frames.items()}  # shallow; touched streams are copied below
    streams = sorted({RAW_COLUMNS[c][0] for c in spec.columns})
    if len(streams) != 1:
        raise ValueError("an injection must stay within one stream")
    stream = streams[0]
    df = frames[stream].copy()
    t = _t_s(df)
    in_iv = (t >= spec.t_start) & (t <= spec.t_end)
    idx = np.flatnonzero(in_iv)
    rec: Dict[str, Any] = {"stream": stream, "affected_raw_rows": int(len(idx))}
    if len(idx) == 0:
        raise ValueError("injection interval contains no raw rows")

    def col_sigma(c: str) -> float:
        return float(sigma[_SIGMA_FEATURE[c]])

    if spec.type == "dropout":
        df = df.loc[~in_iv].reset_index(drop=True)
        rec["removed_rows"] = int(len(idx))
    else:
        for c in spec.columns:
            src = [x for x in df.columns if x.lower() == c.lower()][0]
            v = pd.to_numeric(df[src], errors="coerce").to_numpy(dtype="float64").copy()
            a = spec.amplitude_sigma * col_sigma(c)
            if spec.type == "spike":
                v[idx[: spec.n_samples]] += a
                rec["affected_raw_rows"] = int(min(spec.n_samples, len(idx)))
            elif spec.type in ("multi_feature",):
                v[idx] += a
            elif spec.type == "drift":
                v[idx] += a * np.linspace(0.0, 1.0, len(idx))
            elif spec.type == "stuck":
                v[idx] = v[idx[0]]
            elif spec.type == "missing_values":
                v[idx] = np.nan
            elif spec.type == "oscillation":
                v[idx] += a * np.sin(2 * np.pi * spec.osc_hz * (t[idx] - spec.t_start))
            df[src] = v
    out[stream] = df
    rec["affected_raw_columns"] = list(spec.columns)
    rec["affected_model_features"] = sorted({f for c in spec.columns for f in RAW_COLUMNS[c][1]})
    if spec.type == "spike":
        rec["injected_raw_interval_s"] = [float(t[idx[0]]), float(t[idx[min(spec.n_samples, len(idx)) - 1]])]
    else:
        rec["injected_raw_interval_s"] = [float(t[idx[0]]), float(t[idx[-1]])]
    return out, rec


def _overlaps(w: pd.DataFrame, t0: float, t1: float) -> pd.Series:
    return (w["t_start"] <= t1) & (w["t_end"] >= t0)


def preservation(
    base_aligned: pd.DataFrame,
    new_aligned: pd.DataFrame,
    spec: InjectionSpec,
    rec: Mapping[str, Any],
    sigma: Mapping[str, float],
) -> Dict[str, Any]:
    """Did the pattern survive resampling onto the 10 Hz grid?"""
    t0, t1 = rec["injected_raw_interval_s"]
    feats = rec["affected_model_features"]
    m_b = (base_aligned["_t_s"] >= t0) & (base_aligned["_t_s"] <= t1)
    m_n = (new_aligned["_t_s"] >= t0) & (new_aligned["_t_s"] <= t1)
    n_grid = int(m_n.sum())
    out: Dict[str, Any] = {"grid_samples_in_interval": n_grid}
    if spec.type in ("missing_values", "dropout"):
        sub = new_aligned.loc[m_n, feats]
        gap = int((new_aligned.loc[m_n, "_session_id"] < 0).sum())
        nonfinite = int((~np.isfinite(sub.to_numpy(dtype=float))).any(axis=1).sum())
        out.update(grid_samples_nonfinite=nonfinite, grid_samples_in_gap=gap,
                   pattern_preserved=bool(nonfinite > 0 or gap > 0), max_delta_sigma=None)
        return out
    if n_grid == 0:
        return {**out, "pattern_preserved": False, "max_delta_sigma": 0.0,
                "note": "no 10 Hz grid sample falls inside the injected interval"}
    if spec.type == "stuck":
        res = []
        for f in feats:
            b = base_aligned.loc[m_b, f].to_numpy(dtype=float)
            n = new_aligned.loc[m_n, f].to_numpy(dtype=float)
            sb, sn = np.nanstd(b) / sigma[f], np.nanstd(n) / sigma[f]
            res.append((sb, sn))
        base_sig = max(r[0] for r in res)
        vac = base_sig < VACUOUS_BASELINE_SIGMA
        out.update(
            baseline_std_sigma=float(base_sig),
            injected_std_sigma=float(max(r[1] for r in res)),
            vacuous=bool(vac),
            max_delta_sigma=None,
            pattern_preserved=bool((not vac) and all(r[1] <= STUCK_RESIDUAL_FRACTION * max(r[0], 1e-12) for r in res)),
        )
        return out
    deltas = []
    for f in feats:
        b = base_aligned.set_index("_t_s")[f]
        n = new_aligned.set_index("_t_s")[f]
        common = n.index[(n.index >= t0) & (n.index <= t1)].intersection(b.index)
        d = np.abs(n.loc[common].to_numpy(dtype=float) - b.loc[common].to_numpy(dtype=float)) / sigma[f]
        if len(d):
            deltas.append(float(np.nanmax(d)))
    mx = max(deltas) if deltas else 0.0
    out.update(max_delta_sigma=mx, nominal_amplitude_sigma=spec.amplitude_sigma,
               pattern_preserved=bool(mx >= PRESERVE_FRACTION * spec.amplitude_sigma))
    return out


def sigma_by_feature(artifacts: ModelArtifacts) -> Dict[str, float]:
    return {f: float(s) for f, s in zip(artifacts.features, artifacts.scaler["std"])}


def plan_injection(
    baseline_windows: pd.DataFrame,
    itype: str,
    level: Mapping[str, Any],
    rng: np.random.Generator,
    seed: int,
    injection_id: str,
    threshold: float,
) -> Optional[InjectionSpec]:
    """Pick a clean placement: one session, every overlapping baseline window unflagged."""
    dur = float(level["duration_s"])
    w = baseline_windows
    cands = []
    for s, g in w.groupby("session_id"):
        s_lo, s_hi = float(g["t_start"].min()), float(g["t_end"].max())
        for t0 in np.unique(g["t_start"].to_numpy()):
            t1 = float(t0) + dur
            if t1 > s_hi - 0.5 or t0 < s_lo + 0.5:
                continue
            ov = w[_overlaps(w, t0, t1)]
            if len(ov) and (ov["anomaly_score"] <= threshold).all() and (ov["session_id"] == s).all():
                cands.append(float(t0))
    if not cands:
        return None
    t0 = float(rng.choice(np.asarray(cands)))
    if itype == "multi_feature":
        cols = [str(c) for c in rng.choice(["GyrX", "GyrY", "GyrZ", "AccX", "AccY", "AccZ"], size=3, replace=False)]
    elif itype == "dropout":
        cols = ["GyrX"]  # stream-level: IMU rows removed
    else:
        cols = [str(rng.choice(list(RAW_COLUMNS)))]
    return InjectionSpec(
        injection_id=injection_id, type=itype, columns=cols,
        t_start=t0, t_end=t0 + dur,
        amplitude_sigma=float(level.get("amplitude_sigma", 0.0)),
        n_samples=int(level.get("n_samples", 3)), seed=seed,
    )


DEFAULT_LEVELS: Dict[str, List[Dict[str, Any]]] = {
    "spike": [{"amplitude_sigma": 3.0, "duration_s": 1.0, "n_samples": 3}, {"amplitude_sigma": 8.0, "duration_s": 1.0, "n_samples": 3}],
    "drift": [{"amplitude_sigma": 3.0, "duration_s": 6.0}, {"amplitude_sigma": 8.0, "duration_s": 6.0}],
    "stuck": [{"duration_s": 3.0}, {"duration_s": 6.0}],
    "missing_values": [{"duration_s": 1.0}, {"duration_s": 4.0}],
    "dropout": [{"duration_s": 1.0}, {"duration_s": 4.0}],
    "oscillation": [{"amplitude_sigma": 3.0, "duration_s": 4.0}, {"amplitude_sigma": 8.0, "duration_s": 4.0}],
    "multi_feature": [{"amplitude_sigma": 3.0, "duration_s": 3.0}, {"amplitude_sigma": 8.0, "duration_s": 3.0}],
}


def run_case(
    frames: Mapping[str, pd.DataFrame],
    spec: InjectionSpec,
    *,
    flight_id: str,
    base: Mapping[str, Any],
    artifacts: ModelArtifacts,
    detector: UAVDetector,
    sigma: Mapping[str, float],
) -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame]]:
    new_frames, rec = apply_injection(frames, spec, sigma)
    res = score_flight_frames(new_frames, artifacts, detector, flight_id)
    w, bw = res["windows"], base["windows"]
    t0, t1 = rec["injected_raw_interval_s"]
    pres = preservation(base["aligned"], res["aligned"], spec, rec, sigma)
    ov = w[_overlaps(w, t0, t1)] if len(w) else w
    bov = bw[_overlaps(bw, t0, t1)] if len(bw) else bw
    # windows that would have been scored at baseline but were lost to the injection
    lost = int(len(bov) - len(ov)) if len(bov) > len(ov) else 0
    thr = artifacts.threshold
    flagged_ov = ov[ov["anomaly_score"] > thr]
    record = {
        **spec.to_dict(),
        "source_flight": flight_id,
        **rec,
        **{f"preservation_{k}": v for k, v in pres.items()},
        "pattern_preserved": pres["pattern_preserved"],
        "windows_overlapping_baseline": int(len(bov)),
        "windows_overlapping_scored": int(len(ov)),
        "windows_lost_to_injection": lost,
        "eligible_for_scoring": bool(len(ov) > 0),
        "baseline_max_overlap_score": float(bov["anomaly_score"].max()) if len(bov) else None,
        "max_overlap_score": float(ov["anomaly_score"].max()) if len(ov) else None,
        "overlap_windows": [
            {"t_start": float(r.t_start), "t_end": float(r.t_end), "score": float(r.anomaly_score)}
            for r in ov.itertuples()
        ],
        "detected": bool(len(flagged_ov) > 0),
        "latency_s": (
            max(0.0, float(flagged_ov["t_end"].min()) - t0) if len(flagged_ov) else None
        ),
    }
    # collateral: among windows present in both runs (same start/end) and outside the
    # injected interval, how many changed flag? Windows that exist only in the injected
    # run (window phase shifts after a source-gap session split) are counted separately.
    key = ["t_start", "t_end"]
    merged = w.merge(bw[key + ["is_anomaly"]], on=key, how="left", suffixes=("", "_base"))
    out_iv = ~_overlaps(merged, t0, t1)
    matched = out_iv & merged["is_anomaly_base"].notna()
    record["collateral_flag_changes_outside_interval"] = int(
        (merged.loc[matched, "is_anomaly"].astype(bool) != merged.loc[matched, "is_anomaly_base"].astype(bool)).sum()
    )
    record["windows_outside_interval_without_baseline_counterpart"] = int((out_iv & ~matched).sum())
    fl_un = merged.loc[out_iv & ~matched & merged["is_anomaly"].astype(bool)]
    record["flagged_windows_outside_interval_without_baseline_counterpart"] = int(len(fl_un))
    base_fl = bw[bw["is_anomaly"].astype(bool)]
    record["flagged_unmatched_not_explained_by_baseline_flags"] = int(sum(
        not ((base_fl["t_start"] <= r.t_end) & (base_fl["t_end"] >= r.t_start)).any()
        for r in fl_un.itertuples()
    ))
    return record, new_frames


def detection_at_threshold(records: Sequence[Mapping[str, Any]], tau: float) -> Dict[str, Any]:
    """Injection-event-level detection at threshold ``tau`` (see module docstring)."""
    n = len(records)
    elig = [r for r in records if r["eligible_for_scoring"]]
    conf = [r for r in elig if (r["baseline_max_overlap_score"] or 0.0) > tau]
    ok = [r for r in elig if r not in conf]
    det = [r for r in ok if (r["max_overlap_score"] or -1.0) > tau]
    lat = []
    for r in det:
        fl = [w for w in r["overlap_windows"] if w["score"] > tau]
        lat.append(max(0.0, min(w["t_end"] for w in fl) - r["t_start"]))
    return {
        "threshold": float(tau),
        "n_injections": n,
        "n_eligible": len(elig),
        "n_not_eligible": n - len(elig),
        "n_confounded_by_baseline": len(conf),
        "n_evaluable": len(ok),
        "n_detected": len(det),
        "detection_rate_eligible": (len(det) / len(ok)) if ok else None,
        "detection_rate_all_injections": (len(det) / n) if n else None,
        "latency_s_median": float(np.median(lat)) if lat else None,
        "latency_s_max": float(np.max(lat)) if lat else None,
        "missed": [r["injection_id"] for r in ok if r not in det],
    }


def summarize(records: Sequence[Mapping[str, Any]], tau: float) -> Dict[str, Any]:
    """Overall and per-type/per-preservation detection at ``tau``."""
    by_type = {}
    for t in INJECTION_TYPES:
        rs = [r for r in records if r["type"] == t]
        if rs:
            by_type[t] = detection_at_threshold(rs, tau)
    pres = [r for r in records if r["pattern_preserved"]]
    npres = [r for r in records if not r["pattern_preserved"]]
    return {
        "overall": detection_at_threshold(records, tau),
        "by_type": by_type,
        "pattern_preserved": {"n": len(pres), **detection_at_threshold(pres, tau)} if pres else {"n": 0},
        "pattern_not_preserved": {"n": len(npres), **detection_at_threshold(npres, tau)} if npres else {"n": 0},
    }
