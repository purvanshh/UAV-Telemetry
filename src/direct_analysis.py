"""Deterministic summarisation for *Direct AI Telemetry Analysis* of a single CSV.

Everything here is plain Python/numpy: no model, no LSTM, no scaler, no threshold. It turns a telemetry
table into bounded, evidence-linked summaries that a language model can describe and an operator can
check against the source rows.

Principles
----------
* **Unit-free detection.** Excursions, offsets, flat-lines and gaps are found with robust statistics in
  each channel's own raw units (and the time axis in its own raw ticks). No unit conversion, sign flip or
  coordinate transform is applied. Where a time unit is only hypothesised, seconds are shown as
  "assuming ..." and flagged.
* **Traceable.** Every event carries the original data-row numbers and the raw timestamp values of its
  first and last sample, and measurements with stable evidence ids (``E1.gx.max``) that the report
  validator resolves.
* **Bounded.** Row, column, cell and wall-clock limits; fixed-size series envelopes.

Not implemented (and therefore never reported): oscillation detection and commanded-vs-measured
tracking, whose signal semantics are unverified.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.combined_ingest import (
    DETERMINISTIC_TIME,
    TIME_HINTS,
    CombinedFileError,
    Inspection,
    inspect_combined,
)
from src.dataset_profiles import Profile, ProfileMatch, match_profile_detailed
from src.operator_report import build_operator_report
from src.pilot_report import friendly_signal_label
from src.event_engine import build_events, mad_sigma, r6, runs

ANALYSIS_VERSION = "direct-2"

DEFAULT_MAX_ROWS = 500_000
DEFAULT_MAX_COLUMNS = 100
DEFAULT_MAX_CELLS = 20_000_000
DEFAULT_TIME_BUDGET_S = 20.0

SERIES_BINS = 40
EVENT_POINTS = 24
MAX_SERIES_CHANNELS = 14
GAP_FACTOR = 5.0  # a step this many times the median step is a gap (unit-free)
FLAG_MAX_CHANGES = 12
MAX_PEAKS = 14


class ProcessingLimit(ValueError):
    """The file exceeds a documented resource limit (rows, columns, cells or time budget)."""


def limits() -> Dict[str, Any]:
    def f(name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, default))
        except ValueError:
            return float(default)

    return {
        "max_rows": int(f("UAV_DIRECT_MAX_ROWS", DEFAULT_MAX_ROWS)),
        "max_columns": int(f("UAV_DIRECT_MAX_COLUMNS", DEFAULT_MAX_COLUMNS)),
        "max_cells": int(f("UAV_DIRECT_MAX_CELLS", DEFAULT_MAX_CELLS)),
        "time_budget_s": f("UAV_DIRECT_TIME_BUDGET_S", DEFAULT_TIME_BUDGET_S),
    }


class Budget:
    def __init__(self, seconds: float):
        self.t0, self.seconds = time.monotonic(), seconds

    def check(self, where: str = "") -> None:
        if time.monotonic() - self.t0 > self.seconds:
            raise ProcessingLimit(f"Analysis exceeded its {self.seconds:.0f} s processing budget ({where}).")


# ---- channel metadata ---------------------------------------------------------------------------------------------

GROUP_OF_CANONICAL = {
    "roll": "attitude", "pitch": "attitude", "acc_x": "accelerometer", "acc_y": "accelerometer", "acc_z": "accelerometer",
    "gyro_x": "gyroscope", "gyro_y": "gyroscope", "gyro_z": "gyroscope",
}
ROLE_GROUP = {"command": "pilot_command", "motor": "motor_output", "control": "controller_internal",
              "distance": "distance_height", "vertical": "distance_height", "status": "status_flag"}


@dataclass
class Channel:
    name: str
    group: str = "unclassified"
    role: Optional[str] = None
    canonical: Optional[str] = None
    unit: Optional[str] = None  # unit only if VERIFIED
    unit_hypothesis: Optional[str] = None
    unit_status: str = "unknown"  # verified | supported | unverified | unknown
    meaning_status: str = "unverified"  # profile semantic status
    handling: str = ""
    is_flag: bool = False
    numeric: bool = True

    def public(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in ("name", "group", "role", "canonical", "unit", "unit_hypothesis",
                                               "unit_status", "meaning_status", "handling", "is_flag")}


def _classify(headers: List[str], profile: Optional[Profile]) -> Dict[str, Channel]:
    out: Dict[str, Channel] = {}
    for h in headers:
        ch = Channel(name=h)
        pc = profile.column_for(h) if profile else None
        if pc is not None:
            ch.role, ch.canonical = pc.role, pc.canonical
            ch.group = GROUP_OF_CANONICAL.get(pc.canonical or "", ROLE_GROUP.get(pc.role, "unclassified"))
            ch.meaning_status = pc.status
            ch.handling = pc.handling
            ch.unit_status = pc.unit_status if (pc.unit or pc.unit_hypothesis) else "unknown"
            if pc.unit_status == "verified":
                ch.unit = pc.unit
            else:
                ch.unit_hypothesis = pc.unit or pc.unit_hypothesis
        out[h] = ch
    return out


# ---- time axis ------------------------------------------------------------------------------------------------------

def _find_time(headers: List[str], profile: Optional[Profile]) -> Tuple[Optional[str], Dict[str, Any]]:
    norm = lambda s: re.sub(r"[\s_]+", "", s).lower()  # noqa: E731
    if profile and profile.time_column:
        col = next((h for h in headers if norm(h) == norm(profile.time_column)), None)
        if col:
            st = profile.time_unit_status
            return col, {"column": col, "unit_hypothesis": profile.time_unit, "status": st,
                         "source": f"profile {profile.label}", "evidence": profile.time_evidence}
    for h in headers:
        if norm(h) in DETERMINISTIC_TIME:
            return h, {"column": h, "unit_hypothesis": DETERMINISTIC_TIME[norm(h)][0], "status": "verified",
                       "source": "dataset column name", "evidence": ""}
    for h in headers:
        if norm(h) in TIME_HINTS:
            return h, {"column": h, "unit_hypothesis": TIME_HINTS[norm(h)], "status": "unverified",
                       "source": "column name only", "evidence": ""}
    return None, {"column": None, "unit_hypothesis": None, "status": "none", "source": "no time column",
                  "evidence": "", "note": "Positions are reported as data-row numbers; durations are unavailable."}


_UNIT_TO_SECONDS = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}


# ---- helpers --------------------------------------------------------------------------------------------------------

_mad_sigma, _r, _runs = mad_sigma, r6, runs


def _stats(y: np.ndarray) -> Dict[str, Any]:
    n = len(y)
    fin = np.isfinite(y)
    f = y[fin]
    base: Dict[str, Any] = {"n": int(n), "n_missing": int(np.isnan(y).sum()), "n_non_finite": int(np.isinf(y).sum())}
    if len(f) == 0:
        return {**base, "n_valid": 0}
    q = np.percentile(f, [1, 5, 25, 50, 75, 95, 99])
    return {**base, "n_valid": int(len(f)), "min": _r(f.min()), "max": _r(f.max()), "mean": _r(f.mean()),
            "std": _r(f.std()), "median": _r(q[3]), "p01": _r(q[0]), "p05": _r(q[1]), "p25": _r(q[2]), "p75": _r(q[4]),
            "p95": _r(q[5]), "p99": _r(q[6]), "first": _r(f[0]), "last": _r(f[-1]),
            "n_unique": int(len(np.unique(f))), "constant": bool(np.ptp(f) == 0)}


# ---- main analysis -------------------------------------------------------------------------------------------------------

@dataclass
class DirectSummary:
    data: Dict[str, Any]
    table: pd.DataFrame  # numeric channels, analysis order, with original row numbers in `__row`
    time_values: Optional[np.ndarray]


def analyze_table(raw: bytes, filename: str) -> DirectSummary:
    lim = limits()
    budget = Budget(lim["time_budget_s"])
    try:
        insp: Inspection = inspect_combined(raw, filename, max_rows=lim["max_rows"], max_columns=lim["max_columns"])
    except CombinedFileError as exc:
        raise ProcessingLimit(str(exc)) from exc
    if len(insp.df) * len(insp.headers) > lim["max_cells"]:
        raise ProcessingLimit(f"The table has {len(insp.df) * len(insp.headers):,} cells; the limit is {lim['max_cells']:,}.")

    match: ProfileMatch = match_profile_detailed(insp.headers)
    profile = match.profile if match.applies else None
    chans = _classify(insp.headers, profile)
    tcol, tinfo = _find_time(insp.headers, profile)

    df = insp.df
    num = pd.DataFrame({h: pd.to_numeric(df[h], errors="coerce") for h in insp.headers})
    text_cols = [h for h in insp.headers if insp.stats[h]["n_text"] > 0 and insp.stats[h]["numeric_fraction"] < 0.99]
    for h in text_cols:
        chans[h].numeric = False
    n_raw = len(num)

    # --- time axis & ordering -------------------------------------------------------------------------------------------
    order = np.arange(n_raw)
    tvals: Optional[np.ndarray] = None
    timing: Dict[str, Any] = {"time_column": tcol, "time_basis": tinfo}
    dq_notes: List[str] = []
    if tcol is not None:
        t_all = num[tcol].to_numpy(float)
        bad_t = ~np.isfinite(t_all)
        timing["rows_without_valid_timestamp"] = int(bad_t.sum())
        if bad_t.sum() == n_raw:
            raise ProcessingLimit(f"Column '{tcol}' has no valid timestamps; it cannot be used as the time axis.")
        steps_file = np.diff(t_all[~bad_t])
        timing["out_of_order_steps"] = int((steps_file < 0).sum())
        timing["repeated_timestamps"] = int((steps_file == 0).sum())
        keep = np.where(~bad_t)[0]
        keep = keep[np.argsort(t_all[keep], kind="stable")]
        order, tvals = keep, t_all[keep]
        if timing["out_of_order_steps"]:
            dq_notes.append(f"{timing['out_of_order_steps']} timestamps were out of order; rows were analysed in time order.")
        if timing["rows_without_valid_timestamp"]:
            dq_notes.append(f"{timing['rows_without_valid_timestamp']} rows without a valid timestamp were excluded.")
        steps = np.diff(tvals)
        pos = steps[steps > 0]
        med = float(np.median(pos)) if len(pos) else float("nan")
        timing["median_step_raw"] = _r(med)
        timing["step_percentiles_raw"] = {k: _r(v) for k, v in zip(("p01", "p25", "p50", "p75", "p99"),
                                                                 np.percentile(pos, [1, 25, 50, 75, 99]))} if len(pos) else {}
        gaps = [] if not len(pos) else [
            {"after_index": int(i), "start_raw": _r(tvals[i]), "end_raw": _r(tvals[i + 1]), "step_raw": _r(steps[i]),
             "step_over_median": _r(steps[i] / med, 2)} for i in np.where(steps > GAP_FACTOR * med)[0]]
        timing["gaps"] = gaps[:50]
        timing["n_gaps"] = len(gaps)
        timing["first_raw"], timing["last_raw"] = _r(tvals[0]), _r(tvals[-1])
        timing["span_raw"] = _r(tvals[-1] - tvals[0])
        unit = tinfo.get("unit_hypothesis")
        if unit in _UNIT_TO_SECONDS:
            sc = _UNIT_TO_SECONDS[unit]
            qualifier = "verified" if tinfo["status"] == "verified" else f"ASSUMING {unit} (unit not verified)"
            timing["duration_s"] = {"value": _r((tvals[-1] - tvals[0]) * sc, 3), "basis": qualifier}
            if len(pos):
                timing["sampling_hz"] = {"value": _r(1.0 / (med * sc), 2), "basis": qualifier}
    else:
        timing["note"] = tinfo["note"]
    table = num.iloc[order].reset_index(drop=True)
    table.insert(0, "__row", order)
    n = len(table)
    budget.check("timing")

    # --- flags / statistics ------------------------------------------------------------------------------------------------
    stats: Dict[str, Any] = {}
    series_ok: List[str] = []
    for h in insp.headers:
        if h == tcol or not chans[h].numeric:
            continue
        y = table[h].to_numpy(float)
        fin = y[np.isfinite(y)]
        n_changes = int((np.diff(fin) != 0).sum()) if len(fin) > 1 else 0
        # a status flag changes rarely; a two-valued channel that toggles constantly is quantised noise, not a state
        chans[h].is_flag = bool(len(fin) and len(np.unique(fin)) <= 4 and np.all(fin == np.round(fin)) and n_changes <= FLAG_MAX_CHANGES)
        stats[h] = _stats(y)
        stats[h]["is_flag"] = chans[h].is_flag
        if stats[h].get("n_valid", 0) > 1 and not stats[h]["constant"]:
            series_ok.append(h)
        budget.check(f"stats:{h}")

    derived: Dict[str, np.ndarray] = {}
    groups: Dict[str, List[str]] = {}
    for h, c in chans.items():
        if c.numeric and h != tcol:
            groups.setdefault(c.group, []).append(h)
    for gname, label in (("accelerometer", "accel_magnitude_raw"), ("gyroscope", "gyro_magnitude_raw")):
        cols = [c for c in groups.get(gname, []) if c in table]
        if len(cols) == 3:
            derived[label] = np.sqrt(np.sum(np.square(table[cols].to_numpy(float)), axis=1))
            stats[label] = _stats(derived[label])
            stats[label]["derived_from"] = cols
            stats[label]["note"] = "Euclidean norm of the three raw columns; no unit conversion or frame transform applied."

    # --- candidate events (see src/event_engine.py for the documented method) ----------------------------------------------------
    chan_info = {h: {"group": c.group, "role": c.role, "is_flag": c.is_flag, "meaning_status": c.meaning_status} for h, c in chans.items()}
    events, event_overflow, engine = build_events(table, tcol, tinfo, chan_info, series_ok, timing.get("gaps", []), _UNIT_TO_SECONDS,
                                                  check=budget.check)
    transitions: List[Dict[str, Any]] = []
    for h in [c for c in stats if stats[c].get("is_flag")]:
        y = table[h].to_numpy(float)
        ok = np.isfinite(y)
        idx = np.where(ok[1:] & ok[:-1] & (np.diff(np.where(ok, y, 0)) != 0))[0]
        for i in idx[:10]:
            transitions.append({"channel": h, "from": _r(y[i]), "to": _r(y[i + 1]), "index": int(i + 1)})
    budget.check("events")

    peaks = _peaks(table, tcol, chans, series_ok, tinfo)

    # --- correlations (unit-free) ------------------------------------------------------------------------------------------------
    corr = _correlations(table, chans, series_ok, groups)

    # --- phases from a status flag ---------------------------------------------------------------------------------------------
    phases = _phases(table, stats, chans, tcol, derived)

    # --- data quality ------------------------------------------------------------------------------------------------------------
    dq = {
        "rows": int(n_raw), "rows_analysed": int(n), "columns": len(insp.headers), "numeric_columns": len(stats),
        "non_numeric_columns": text_cols,
        "constant_channels": [h for h, s in stats.items() if s.get("constant")],
        "channels_with_missing": {h: {"missing": s["n_missing"], "non_finite": s["n_non_finite"]} for h, s in stats.items()
                                  if s["n_missing"] or s["n_non_finite"]},
        "notes": dq_notes,
    }
    # periods the data cannot speak for: sampling gaps and long runs of missing values (a mixed event is kept as an event)
    unassessed = [{"kinds": e["kinds"], "channels": e["channels"], "start_row": e["start_row"], "end_row": e["end_row"],
                   "start_raw": e["start_raw"], "end_raw": e["end_raw"]}
                  for e in events if e["kinds"] and set(e["kinds"]) & {"sampling_gap", "missing_values"}]

    # --- bounded series -------------------------------------------------------------------------------------------------------------
    key = _key_channels(chans, series_ok)
    series = _envelope(table, tcol, key)
    budget.check("series")

    profile_info = {
        "id": profile.id if profile else None, "version": profile.version if profile else None,
        "match": match.kind, "file_fingerprint": match.fingerprint,
        "profile_fingerprint": profile.fingerprint if profile else None,
        "missing_required": match.missing_required, "extra_columns": match.extra_columns,
        "documentation_status": profile.documentation_status if profile else None,
    }
    limitations = _limitations(chans, profile, tinfo, headers=insp.headers)
    data = {
        "analysis_version": ANALYSIS_VERSION, "filename": filename, "file_sha256": insp.sha256, "size_bytes": insp.size_bytes,
        "profile": profile_info, "timing": timing, "channels": {h: c.public() for h, c in chans.items()},
        "stats": stats, "events": events, "event_overflow": event_overflow, "engine": engine,
        "flag_transitions": transitions[:40], "peaks": peaks, "phases": phases, "correlations": corr, "data_quality": dq,
        "unassessed_periods": unassessed[:30], "series": series, "limitations": limitations,
        "not_implemented": ["oscillation detection", "commanded-vs-measured tracking"],
        "limits": lim,
    }
    data["input_fingerprint"] = fingerprint(data)
    data["operator"] = build_operator_report(data)  # derived from the fingerprinted fields; deterministic
    return DirectSummary(data=data, table=table, time_values=tvals)


def fingerprint(data: Dict[str, Any]) -> str:
    """Stable hash of the deterministic summary (what the model is allowed to see)."""
    core = {k: data[k] for k in ("file_sha256", "profile", "timing", "stats", "events", "peaks", "phases", "series", "analysis_version")}
    return hashlib.sha256(json.dumps(core, sort_keys=True, default=str).encode()).hexdigest()


# ---- evidence ids -------------------------------------------------------------------------------------------------------------------

def evidence_catalog(data: Dict[str, Any]) -> Dict[str, Any]:
    """id -> value for every citable number: event measurements, channel stats, timing."""
    cat: Dict[str, Any] = {}
    for ev in data["events"]:
        for k in ("start_raw", "end_raw", "span_raw", "start_row", "end_row", "n_samples"):
            if ev.get(k) is not None:
                cat[f"{ev['id']}.{k}"] = ev[k]
        for m in ev["measurements"]:
            for k, eid in m["evidence"].items():
                if m.get(k) is not None:
                    cat[eid] = m[k]
    for p in data.get("peaks", []):
        for k in ("raw_time", "value", "channel_median", "departure", "channel_robust_sigma", "row"):
            if p.get(k) is not None:
                cat[f"{p['id']}.{k}"] = p[k]
    for ph in data.get("phases", []):
        for k in ("start_row", "end_row", "start_raw", "end_raw", "n", "accel_magnitude_raw_median"):
            if ph.get(k) is not None:
                cat[f"phase{int(ph['value'])}.{k}"] = ph[k]
    for h, s in data["stats"].items():
        for k, v in s.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                cat[f"stat.{h}.{k}"] = v
    for k, v in data["timing"].items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            cat[f"timing.{k}"] = v
        elif isinstance(v, dict) and isinstance(v.get("value"), (int, float)):
            cat[f"timing.{k}"] = v["value"]
    for k, v in data["data_quality"].items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            cat[f"dq.{k}"] = v
    return cat


def _peaks(table: pd.DataFrame, tcol: Optional[str], chans: Dict[str, Channel], ok: List[str], tinfo: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Largest departure from each key channel's own median: where it happened, and how big, in raw units."""
    out = []
    for h in _key_channels(chans, ok)[:MAX_PEAKS]:
        y = table[h].to_numpy(float)
        fin = np.isfinite(y)
        if fin.sum() < 10:
            continue
        med = float(np.median(y[fin]))
        dev = np.where(fin, np.abs(y - med), -1)
        i = int(np.argmax(dev))
        out.append({"id": f"P.{h}", "channel": h, "index": i, "row": int(table["__row"].iloc[i]),
                    "raw_time": _r(table[tcol].iloc[i]) if tcol else None, "value": _r(y[i]), "channel_median": _r(med),
                    "departure": _r(y[i] - med), "channel_robust_sigma": _r(_mad_sigma(y))})
    return out


# ---- correlations / phases / series ------------------------------------------------------------------------------------------

def _correlations(table: pd.DataFrame, chans: Dict[str, Channel], ok: List[str], groups: Dict[str, List[str]]) -> Dict[str, Any]:
    cand = [h for h in ok if not chans[h].is_flag][:20]
    pairs = []
    if len(cand) >= 2:
        m = table[cand].replace([np.inf, -np.inf], np.nan).corr().to_numpy()
        for i in range(len(cand)):
            for j in range(i + 1, len(cand)):
                if np.isfinite(m[i, j]) and abs(m[i, j]) >= 0.6:
                    pairs.append({"a": cand[i], "b": cand[j], "r": _r(m[i, j], 3)})
        pairs.sort(key=lambda p: -abs(p["r"]))
    rate_pairs = []
    # unit-free: correlation of per-sample change of an attitude column with the matching gyro column
    for att, gyr in (("phi_deg", "gx"), ("theta_deg", "gy")):
        if att in table and gyr in table:
            d = table[att].diff().to_numpy(float)[1:]
            g = table[gyr].to_numpy(float)[1:]
            ok_ = np.isfinite(d) & np.isfinite(g)
            if ok_.sum() > 30 and np.std(d[ok_]) > 0 and np.std(g[ok_]) > 0:
                rate_pairs.append({"a": f"change per sample of {att}", "b": gyr,
                                   "r": _r(np.corrcoef(d[ok_], g[ok_])[0, 1], 3)})
    return {"pairs": pairs[:12], "rate_pairs": rate_pairs,
            "note": "Pearson r on raw columns; correlation, not causation. Pairs with |r| >= 0.6 only."}


def _phases(table: pd.DataFrame, stats: Dict[str, Any], chans: Dict[str, Channel], tcol: Optional[str],
            derived: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    flag = "armed" if "armed" in table and stats.get("armed", {}).get("is_flag") else None
    if not flag:
        return []
    y = table[flag].to_numpy(float)
    out = []
    keys = [h for h in ("phi_deg", "theta_deg", "ax", "ay", "az", "gx", "gy", "gz", "tof_mm") if h in table][:9]
    start = 0
    for i in range(1, len(y) + 1):
        if i == len(y) or y[i] != y[start]:
            if len(out) < 8:
                seg = table.iloc[start:i].replace([np.inf, -np.inf], np.nan)
                ph = {"flag": flag, "value": _r(y[start]), "start_index": start, "end_index": i - 1,
                      "start_row": int(seg["__row"].iloc[0]), "end_row": int(seg["__row"].iloc[-1]),
                      "start_raw": _r(seg[tcol].iloc[0]) if tcol else None, "end_raw": _r(seg[tcol].iloc[-1]) if tcol else None,
                      "n": int(i - start),
                      "median": {h: _r(seg[h].median()) for h in keys}, "std": {h: _r(seg[h].std()) for h in keys}}
                if "accel_magnitude_raw" in derived:
                    ph["accel_magnitude_raw_median"] = _r(np.median(derived["accel_magnitude_raw"][start:i]))
                out.append(ph)
            start = i
    return out


def _key_channels(chans: Dict[str, Channel], ok: List[str]) -> List[str]:
    pri = ["attitude", "gyroscope", "accelerometer", "distance_height", "pilot_command", "motor_output"]
    out: List[str] = []
    for g in pri:
        out += [h for h in ok if chans[h].group == g and not chans[h].is_flag]
    out += [h for h in ok if chans[h].group in ("unclassified", "controller_internal") and not chans[h].is_flag]
    return out[:MAX_SERIES_CHANNELS]


def _envelope(table: pd.DataFrame, tcol: Optional[str], key: List[str], bins: int = SERIES_BINS) -> Dict[str, Any]:
    n = len(table)
    if n == 0 or not key:
        return {"bins": 0, "channels": {}}
    b = min(bins, n)
    edges = np.linspace(0, n, b + 1).astype(int)
    t = table[tcol].to_numpy(float) if tcol else table["__row"].to_numpy(float)
    out: Dict[str, Any] = {"bins": int(b), "x": [_r(np.nanmean(t[edges[i] : edges[i + 1]]), 3) for i in range(b)],
                           "x_is": "raw time ticks" if tcol else "data row number", "channels": {}}
    for h in key:
        y = table[h].to_numpy(float)
        mins, maxs = [], []
        for i in range(b):
            seg = y[edges[i] : edges[i + 1]]
            seg = seg[np.isfinite(seg)]
            mins.append(_r(seg.min(), 4) if len(seg) else None)
            maxs.append(_r(seg.max(), 4) if len(seg) else None)
        out["channels"][h] = {"min": mins, "max": maxs}
    return out


def _limitations(chans: Dict[str, Channel], profile: Optional[Profile], tinfo: Dict[str, Any], headers: List[str]) -> List[str]:
    lim: List[str] = []
    if tinfo.get("status") == "verified":
        pass
    elif tinfo.get("column"):
        lim.append(f"The unit of the time column '{tinfo['column']}' is not verified (hypothesis: {tinfo.get('unit_hypothesis')}). "
                   "Durations in seconds are conditional on that assumption.")
    else:
        lim.append("The file has no recognisable time column, so timing is reported as data-row numbers only.")
    unres = [c.name for c in chans.values() if c.numeric and c.group in ("attitude", "accelerometer", "gyroscope") and c.unit_status != "verified"]
    if unres:
        lim.append("Units are not verified for: " + ", ".join(unres) + ". Values are described in raw units; no conversion was applied.")
    if any(c.group in ("accelerometer", "gyroscope", "attitude") for c in chans.values()):
        lim.append("Axis and sign conventions (roll/pitch direction, accelerometer polarity, gyroscope axes) are not established, "
                   "so no physical direction of motion is stated.")
    low = {h.lower() for h in headers}
    missing = [n for n, alts in (("measured yaw", ("yaw", "yaw_deg", "psi_deg")), ("desired roll/pitch", ("des_roll", "desroll", "phi_sp", "theta_sp")),
                                 ("barometric altitude/pressure/temperature", ("alt", "press", "temp", "baro"))) if not (low & set(alts))]
    if missing:
        lim.append("Not present in this file: " + ", ".join(missing) + ". Nothing was inferred or invented in their place.")
    if any(c.group == "pilot_command" for c in chans.values()):
        lim.append("Pilot/radio command channels are inputs, not measured aircraft state.")
    if any(c.group in ("motor_output", "controller_internal", "status_flag") and c.meaning_status != "verified" for c in chans.values()):
        lim.append("Motor, controller and status columns have unverified meanings and are described by name and raw values only.")
    lim.append("These are statistical observations on raw columns. They are not validated by the deployed anomaly detector and do not establish a fault, "
               "a cause, or that the flight was safe or unsafe.")
    return lim


# ---- chart context for the UI ---------------------------------------------------------------------------------------------------------

def _decimate(idx: np.ndarray, y: np.ndarray, max_points: int) -> np.ndarray:
    """Keep each bucket's minimum and maximum (in time order) so brief spikes survive down-sampling."""
    if len(idx) <= max_points:
        return idx
    nb = max(1, max_points // 2)
    keep = {int(idx[0]), int(idx[-1])}
    for chunk in np.array_split(np.arange(len(idx)), nb):
        seg = y[idx[chunk]]
        if np.isfinite(seg).any():
            keep.add(int(idx[chunk][int(np.nanargmin(seg))]))
            keep.add(int(idx[chunk][int(np.nanargmax(seg))]))
    return np.array(sorted(keep))


def event_context(summary: Dict[str, Any], table: pd.DataFrame, event_id: str, max_points: int = 400) -> Dict[str, Any]:
    """Chart data for one event: pre-event baseline window, the event interval and a post-event window, per channel.

    The horizontal axis is raw time ticks (or data rows if there is no time column) because the time unit is usually
    unresolved; seconds are offered only as a conditional label. Each channel carries its own before/during/after
    statistics so the UI can show "Unavailable" instead of an empty or zero value.
    """
    ev = next((e for e in summary["events"] if e["id"] == event_id), None)
    if ev is None:
        raise KeyError(event_id)
    n = len(table)
    s, e = ev["start_index"], ev["end_index"]
    w = int((summary.get("engine") or {}).get("baseline_window_samples") or max(25, (e - s)))
    lo, hi = max(0, s - w), min(n - 1, e + w)
    tcol = summary["timing"].get("time_column")
    tinfo = summary["timing"]["time_basis"]
    xs = table[tcol].to_numpy(float) if tcol else table["__row"].to_numpy(float)
    unit = tinfo.get("unit_hypothesis")
    if tcol:
        axis = f"raw time ticks of '{tcol}'" + ("" if tinfo.get("status") == "verified" else f" (unit not verified; hypothesis: {unit})")
        x_is = "raw_ticks"
    else:
        axis, x_is = "data row", "row"
    b_av = s - lo >= 20
    p_av = hi - e >= 10
    regions = {"baseline": {"start": float(xs[lo]), "end": float(xs[max(lo, s - 1)]), "available": bool(b_av)},
               "event": {"start": float(xs[s]), "end": float(xs[e])},
               "post": {"start": float(xs[min(n - 1, e + 1)]), "end": float(xs[hi]), "available": bool(p_av)}}
    chs = []
    base_idx = np.arange(lo, hi + 1)
    for m in ev["measurements"][:8]:
        h = m["channel"]
        y = table[h].to_numpy(float)
        idx = _decimate(base_idx, y, max_points)
        chs.append({"signal": h, "label": friendly_signal_label(h, group=m.get("group"), deep=False),
                    "technical_name": h, "unit": None, "group": m.get("group"), "semantics": m.get("semantics"),
                    "time": [float(xs[i]) for i in idx], "values": [_r(y[i], 5) for i in idx],
                    "before": m["before"], "during": m["during"], "after": m["after"], "kinds": m["kinds"],
                    "persistence": m.get("persistence"), "shift_sigma": m.get("shift_sigma")})
    return {"event_id": event_id, "axis": axis, "x_is": x_is, "regions": regions, "channels": chs,
            "window_start": float(xs[lo]), "window_end": float(xs[hi]), "time": [float(xs[i]) for i in base_idx[:: max(1, len(base_idx) // max_points)]],
            "segment_limited": not (b_av and p_av), "baseline_samples": int(w)}
