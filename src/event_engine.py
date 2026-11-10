"""Deterministic event detection for *Direct AI Telemetry Analysis*.

No model, no LSTM, no unit conversion. Everything is measured in each channel's own raw units and in raw
time ticks. Every constant below has a stated purpose and is exercised by ``tests/test_event_engine.py``.

Detection
---------
* **Rolling causal baseline.** For each continuous channel the baseline at sample *i* is the median of the
  preceding ``W`` samples and its spread is the (approximate) MAD of that window, so a change is always
  judged against the signal's *earlier* behaviour, not against the whole recording.
* **Noise-aware thresholds.** A departure counts only if it exceeds ``Z_THRESHOLD`` robust sigmas of the
  window AND ``MIN_REL_CHANGE`` of the channel's robust range AND ``MIN_RESOLUTION_STEPS`` times the
  channel's recording resolution. The sigma is floored by the channel's step noise, its resolution and a
  fraction of its range, so constant or finely quantised signals cannot produce divide-by-zero "infinite"
  scores. A channel that never changes is skipped.
* **Persistence.** A departure lasting ``<= SPIKE_MAX_SAMPLES`` is a *spike* (and needs
  ``SPIKE_Z_FACTOR`` times more evidence); a longer one is a *sustained shift* if the level after it is
  still away from the earlier baseline (or the file ends first), otherwise a *transient excursion*.
* **Variability.** Block-wise robust spread compared with the preceding blocks (``VAR_RATIO``).
* **Separation.** Findings are grouped into three categories that are never merged with each other:
  ``data_quality`` (missing values, flat-lines, sampling gaps), ``state_transition`` (flag/status channels
  and pilot-command channels, which are inputs and expected to move) and ``statistical_irregularity``
  (everything else, including channels whose meaning is unverified). Related events in other categories
  are linked as ``concurrent_with``.
* **Merging.** Within a category, detections whose intervals are within ``tol`` samples are chained into one
  event, capped at ``EVENT_MAX_SPAN_FRACTION`` of the recording so one blob cannot swallow the flight.
* **Cross-channel association.** Onset times are compared: ``simultaneous`` (within ``SIM_TOL`` samples),
  ``sequential`` (separated by more), group-wise co-movement, and Pearson correlation inside the event
  window. These are *observed associations*. ``physical_relationship`` is always "not established".
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---- documented constants -------------------------------------------------------------------------------------------

BASELINE_MIN = 20          # samples of preceding history required before a baseline is "available"
POST_MIN = 10              # samples after an event required before a post-event comparison is "available"
Z_THRESHOLD = 6.0          # departure, in robust sigmas of the preceding window, that counts as a change
SPIKE_Z_FACTOR = 3.0       # an isolated brief departure needs this many times more evidence (guards heavy-tailed noise)
SPIKE_MAX_SAMPLES = 3      # runs this short are spikes (no persistence)
BRIDGE = 3                 # gaps of up to this many samples inside one departure are bridged
MIN_REL_CHANGE = 0.05      # a change must be at least this fraction of the channel's robust range (p99-p1)
MIN_RESOLUTION_STEPS = 3   # ... and at least this many multiples of the recording resolution
SIGMA_RANGE_FLOOR = 0.005  # spread is never taken below this fraction of the range (constant-ish channels)
VAR_RATIO = 4.0            # block spread this many times the preceding blocks' spread = variability increase
VAR_MIN_RANGE = 0.03       # ... and the block's p95-p5 must exceed this fraction of the channel range
SIM_TOL = 3                # onsets within this many samples are "simultaneous"
FLAT_MIN_RUN = 15
MISSING_MIN_RUN = 10
GAP_PRIORITY_MIN = 2
CORR_MIN_ABS = 0.7         # in-event correlation worth mentioning
CORR_MIN_N = 20
EVENT_MAX_SPAN_FRACTION = 0.3
HOLD_WINDOWS = 2           # a shifted level that returns within this many baseline windows is part of the same excursion
MAX_EVENTS = 25
MAX_EVENT_CHANNELS = 24
FLAG_MAX_CHANGES = 12

CATEGORIES = ("data_quality", "state_transition", "statistical_irregularity")


# ---- tiny numeric helpers (shared with src.direct_analysis) ----------------------------------------------------------

def r6(x: Any, nd: int = 6) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if math.isfinite(v) else None


def mad_sigma(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Inclusive (start, end) index pairs of contiguous True runs."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0].tolist(), (np.where(d == -1)[0] - 1).tolist()))


def _bridge(mask: np.ndarray, gap: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for s, e in runs(mask):
        if out and s - out[-1][1] - 1 <= gap:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def baseline_window(n: int) -> int:
    return int(min(100, max(BASELINE_MIN, n // 10)))


# ---- channel noise model ---------------------------------------------------------------------------------------------

def channel_scale(y: np.ndarray) -> Optional[Dict[str, float]]:
    """Robust range, recording resolution and noise floor of one channel; None if it never changes."""
    f = y[np.isfinite(y)]
    if len(f) < 30 or np.ptp(f) == 0:
        return None
    rng = float(np.percentile(f, 99) - np.percentile(f, 1)) or float(np.ptp(f))
    u = np.unique(f)
    gaps = np.diff(u)
    res = float(gaps[gaps > 0].min()) if len(gaps) else 0.0
    d = np.diff(y)
    d = d[np.isfinite(d)]
    step_sigma = mad_sigma(d) / math.sqrt(2) if len(d) else 0.0
    floor = max(step_sigma, res, SIGMA_RANGE_FLOOR * rng)
    return {"range": rng, "resolution": res, "floor": floor}


def rolling_baseline(y: np.ndarray, w: int, floor: float) -> Tuple[np.ndarray, np.ndarray]:
    """Causal rolling median and robust sigma of the *preceding* ``w`` samples (NaN until BASELINE_MIN exist)."""
    s = pd.Series(y)
    med = s.rolling(w, min_periods=BASELINE_MIN).median()
    mad = (s - med).abs().rolling(w, min_periods=BASELINE_MIN).median()
    med = med.shift(1).to_numpy()
    sig = np.maximum(1.4826 * mad.shift(1).to_numpy(), floor)
    return med, sig


# ---- per-channel detectors -------------------------------------------------------------------------------------------

def _window_stats(y: np.ndarray, lo: int, hi: int, min_n: int) -> Dict[str, Any]:
    seg = y[max(0, lo) : max(0, hi)]
    f = seg[np.isfinite(seg)]
    if len(f) < min_n:
        return {"available": False, "n": int(len(f)), "median": None, "min": None, "max": None, "sigma": None}
    dif = np.diff(seg)
    return {"available": True, "n": int(len(f)), "median": r6(np.median(f)), "min": r6(f.min()), "max": r6(f.max()),
            "sigma": r6(mad_sigma(f)), "jitter": r6(mad_sigma(dif[np.isfinite(dif)]) if np.isfinite(dif).sum() > 2 else 0.0)}


def detect_departures(y: np.ndarray, w: int) -> List[Dict[str, Any]]:
    """Spikes, sustained shifts and transient excursions relative to the preceding rolling baseline."""
    sc = channel_scale(y)
    if sc is None:
        return []
    med, sig = rolling_baseline(y, w, sc["floor"])
    with np.errstate(all="ignore"):
        diff = y - med
        z = diff / sig
    absdiff = np.abs(diff)
    ok = np.isfinite(z) & np.isfinite(y)
    big = ok & (absdiff > MIN_REL_CHANGE * sc["range"]) & (absdiff >= MIN_RESOLUTION_STEPS * sc["resolution"])
    out: List[Dict[str, Any]] = []
    for s, e in _bridge(big & (np.abs(z) > Z_THRESHOLD), BRIDGE):
        n_run = e - s + 1
        peak = int(s + np.nanargmax(np.abs(z[s : e + 1])))
        peak_z = float(abs(z[peak]))
        base_med = float(med[s])
        level_end = None
        if n_run <= SPIKE_MAX_SAMPLES:
            if peak_z < SPIKE_Z_FACTOR * Z_THRESHOLD:
                continue
            kind = "spike"
        else:
            # Persistence is judged against the baseline frozen at onset, not the adaptive one (which catches up with a
            # shifted level and would end the run while the level is still shifted).
            lvl_thresh = max(MIN_REL_CHANGE * sc["range"], MIN_RESOLUTION_STEPS * sc["resolution"], 3 * float(sig[s]))
            right_after = y[e + 1 : e + 1 + POST_MIN]
            ra = right_after[np.isfinite(right_after)]
            tail = y[max(s, e - 2) : e + 1]
            probe = ra if len(ra) >= 3 else tail[np.isfinite(tail)]
            still_off = len(probe) > 0 and abs(float(np.median(probe)) - base_med) > lvl_thresh
            if n_run >= 4 and still_off:
                kind = "sustained_shift"
                back = np.abs(y[e + 1 :] - base_med) <= 0.5 * lvl_thresh
                ret = back[:-2] & back[1:-1] & back[2:] if len(back) >= 3 else np.zeros(0, bool)
                hit = np.where(ret)[0]
                level_end = int(e + hit[0]) if len(hit) else len(y) - 1  # where the shifted level ends
            else:
                kind = "transient_excursion"
        # A shift that later returns to its earlier level is reported over the whole excursion (e_level) so the interval matches
        # what the operator sees; a level that never returns is reported as the transition only (merging always uses `e`).
        e_level = level_end if (kind == "sustained_shift" and level_end is not None and level_end < len(y) - 1
                                and level_end - s <= HOLD_WINDOWS * w) else None
        out.append({"kind": kind, "s": int(s), "e": int(e), "onset": int(s), "peak_z": r6(peak_z, 2),
                    "sign": 1 if diff[peak] > 0 else -1, "level_end": level_end, "e_level": e_level})
    return out


def detect_variability(y: np.ndarray, w: int) -> List[Dict[str, Any]]:
    """Block-wise robust spread compared with the three preceding blocks."""
    sc = channel_scale(y)
    b = max(10, w // 2)
    nb = len(y) // b
    if sc is None or nb < 5:
        return []
    blocks = y[: nb * b].reshape(nb, b)
    dif = np.diff(blocks, axis=1)  # sample-to-sample changes inside each block: jitter, not drift of the level
    with np.errstate(all="ignore"):
        med = np.nanmedian(blocks, axis=1)
        spread = np.maximum(1.4826 * np.nanmedian(np.abs(dif - np.nanmedian(dif, axis=1)[:, None]), axis=1), sc["floor"])
        width = np.nanpercentile(blocks, 95, axis=1) - np.nanpercentile(blocks, 5, axis=1)
    flagged = np.zeros(nb, bool)
    for i in range(3, nb):
        base = np.nanmedian(spread[i - 3 : i])
        flagged[i] = bool(np.isfinite(spread[i]) and spread[i] > VAR_RATIO * base and spread[i] > MIN_RESOLUTION_STEPS * sc["resolution"]
                          and width[i] > VAR_MIN_RANGE * sc["range"])
    out: List[Dict[str, Any]] = []
    for s, e in runs(flagged):
        lo, hi = s * b, min(len(y) - 1, (e + 1) * b - 1)
        prior = blocks[max(0, s - 3) : s].ravel()
        base_med = float(np.nanmedian(prior)) if np.isfinite(prior).any() else float(np.nanmedian(y[lo : hi + 1]))
        base_spread = float(np.nanmax([np.nanmedian(spread[max(0, s - 3) : s]), sc["floor"]]))
        # refine the onset from the block boundary to the first sample that clearly leaves the earlier behaviour
        away = np.where(np.isfinite(y[lo : hi + 1]) & (np.abs(y[lo : hi + 1] - base_med) > max(3 * base_spread, MIN_RESOLUTION_STEPS * sc["resolution"])))[0]
        onset = int(lo + away[0]) if len(away) else int(lo)
        out.append({"kind": "variability_increase", "s": onset, "e": int(hi), "onset": onset,
                    "peak_z": r6(float(np.nanmax(spread[s : e + 1])) / base_spread, 2), "sign": 0})
    return out


def detect_flat(y: np.ndarray, is_flag: bool, w: int = 50) -> List[Dict[str, Any]]:
    """A run of exactly repeated values in a channel that was moving just before the run.

    A channel idling at a constant value from the start (e.g. an output at its idle level before it is used) or one that
    repeats values by nature (quantised) is not a flat-line, so the window before the run must show ordinary movement.
    """
    fin = np.isfinite(y)
    if is_flag or fin.sum() < 50:
        return []
    d = np.diff(y)
    ok = np.isfinite(d)
    if ok.sum() == 0 or len(np.unique(y[fin])) < 20:
        return []
    run_min = max(FLAT_MIN_RUN, len(y) // 50)
    out = []
    for s, e in runs(ok & (d == 0)):
        if e - s + 1 < run_min - 1:
            continue
        pre = d[max(0, s - w) : s]
        pre = pre[np.isfinite(pre)]
        if len(pre) >= BASELINE_MIN and (pre != 0).mean() >= 0.7:
            out.append({"kind": "flatline", "s": int(s), "e": int(e + 1), "onset": int(s), "peak_z": None, "sign": 0})
    return out


def detect_missing(y: np.ndarray) -> List[Dict[str, Any]]:
    return [{"kind": "missing_values", "s": int(s), "e": int(e), "onset": int(s), "peak_z": None, "sign": 0}
            for s, e in runs(~np.isfinite(y)) if e - s + 1 >= MISSING_MIN_RUN]


def _cap_keep_groups(meas: List[Dict[str, Any]], cap: int) -> List[Dict[str, Any]]:
    """Keep the strongest channels, but never split a labelled group (e.g. motor-labelled) across the cut."""
    if len(meas) <= cap:
        return meas
    keep = meas[:cap]
    groups = {m["group"] for m in keep if m["group"] not in (None, "unclassified", "status_flag")}
    extra = [m for m in meas[cap:] if m["group"] in groups]
    return keep + extra


# ---- event assembly --------------------------------------------------------------------------------------------------

def _category(kind: str, ch: Dict[str, Any]) -> str:
    if kind in ("missing_values", "flatline", "sampling_gap"):
        return "data_quality"
    if kind == "status_change" or ch.get("is_flag") or ch.get("role") == "command":
        return "state_transition"
    return "statistical_irregularity"


def _kind_for_category(kind: str, cat: str) -> str:
    return "command_change" if (cat == "state_transition" and kind in ("spike", "sustained_shift", "transient_excursion", "variability_increase")) else kind


def _semantics(ch: Dict[str, Any]) -> str:
    if ch.get("is_flag"):
        return "status/flag channel: meaning unverified"
    if ch.get("role") == "command":
        return "pilot command input (not a measured aircraft state)"
    if ch.get("meaning_status") == "verified":
        return "channel with verified meaning"
    return "raw channel: meaning unverified"


def _measure(y: np.ndarray, s: int, e: int, det: List[Dict[str, Any]], ch: Dict[str, Any], name: str,
             w: int, t: Optional[np.ndarray], rows: np.ndarray, rng_ref: float, floor: float = 0.0) -> Optional[Dict[str, Any]]:
    seg = y[s : e + 1]
    fin = seg[np.isfinite(seg)]
    if len(fin) == 0:
        return None
    before = _window_stats(y, s - w, s, BASELINE_MIN)
    after = _window_stats(y, e + 1, e + 1 + w, POST_MIN)
    during = {"available": True, "n": int(len(fin)), "median": r6(np.median(fin)), "min": r6(fin.min()), "max": r6(fin.max()),
              "sigma": r6(mad_sigma(fin)), "jitter": r6(mad_sigma(np.diff(seg)[np.isfinite(np.diff(seg))]) if len(seg) > 3 else 0.0)}
    onset = min(d["onset"] for d in det)
    kinds = sorted({_kind_for_category(d["kind"], _category(d["kind"], ch)) for d in det})
    shift = r6(during["median"] - before["median"]) if before["available"] else None
    sigma_b = before["sigma"] if before["available"] else None
    ref_sigma = max(sigma_b or 0.0, floor)
    shift_sigma = r6(shift / ref_sigma, 2) if (shift is not None and ref_sigma > 0) else None
    if before["available"]:
        peak_dev = float(np.nanmax(np.abs(fin - before["median"])))
        peak_sign = 1 if abs(fin.max() - before["median"]) >= abs(fin.min() - before["median"]) else -1
    else:
        peak_dev, peak_sign = None, 0
    persistence = None
    for d in det:
        if d["kind"] in ("sustained_shift", "transient_excursion", "spike"):
            persistence = {"sustained_shift": "sustained", "transient_excursion": "returned", "spike": "brief"}[d["kind"]]
            break
    lvl_ends = [d["level_end"] for d in det if d.get("level_end") is not None]
    held_until = max(lvl_ends) if lvl_ends else None
    if persistence == "sustained" and held_until is not None:
        persistence = ("sustained: the new level lasted to the end of the recording" if held_until >= len(y) - 1
                       else f"sustained: the new level held until row {int(rows[held_until])}")
    elif persistence == "sustained" and not after["available"]:
        persistence = "sustained (no post-event data to confirm)"
    swing = r6(during["max"] - during["min"])
    m: Dict[str, Any] = {
        "channel": name, "group": ch.get("group"), "role": ch.get("role"), "semantics": _semantics(ch), "kinds": kinds,
        "before": before, "during": during, "after": after,
        "before_median": before["median"], "during_min": during["min"], "during_max": during["max"],
        "during_median": during["median"], "after_median": after["median"],
        "max_step": r6(np.nanmax(np.abs(np.diff(seg)))) if len(seg) > 1 and np.isfinite(np.diff(seg)).any() else None,
        "swing": swing, "shift": shift, "shift_sigma": shift_sigma,
        "peak_departure": r6(peak_dev) if peak_dev is not None else None,
        "peak_z": max((d["peak_z"] for d in det if d.get("peak_z") is not None), default=None),
        "direction": ("higher" if shift > 0 else "lower" if shift < 0 else "unchanged") if shift is not None and abs(shift) > 0 else None,
        "peak_direction": ("higher" if peak_sign > 0 else "lower") if peak_sign else None,
        "persistence": persistence, "level_held_until_row": int(rows[held_until]) if held_until is not None else None,
        "onset_index": int(onset), "onset_row": int(rows[onset]), "onset_raw": r6(t[onset]) if t is not None else None,
        "relative_swing": r6(swing / rng_ref, 3) if rng_ref > 0 and swing is not None else None,
    }
    return m


def build_events(
    table: pd.DataFrame,
    tcol: Optional[str],
    tinfo: Dict[str, Any],
    chans: Dict[str, Dict[str, Any]],
    series_ok: List[str],
    gaps: List[Dict[str, Any]],
    unit_to_seconds: Dict[str, float],
    check: Callable[[str], None] = lambda where: None,
) -> Tuple[List[Dict[str, Any]], int, Dict[str, Any]]:
    """Return (events, overflow_count, engine_info). ``chans`` maps name -> {group, role, is_flag, meaning_status}."""
    n = len(table)
    w = baseline_window(n)
    raw: List[Dict[str, Any]] = []
    for h in series_ok:
        y = table[h].to_numpy(float)
        ch = chans[h]
        found: List[Dict[str, Any]] = []
        if ch.get("is_flag"):
            pass  # status changes are handled below
        else:
            found += detect_departures(y, w) + detect_variability(y, w)
        found += detect_flat(y, bool(ch.get("is_flag")), w) + detect_missing(y)
        for d in found:
            cat = _category(d["kind"], ch)
            raw.append({**d, "channel": h, "category": cat, "s": max(0, d["s"]), "e": min(n - 1, d["e"])})
        check(f"detect:{h}")
    for h in [c for c in series_ok if chans[c].get("is_flag")]:
        y = table[h].to_numpy(float)
        ok = np.isfinite(y)
        idx = np.where(ok[1:] & ok[:-1] & (np.diff(np.where(ok, y, 0)) != 0))[0]
        for i in idx[:10]:
            raw.append({"kind": "status_change", "channel": h, "category": "state_transition", "s": int(i), "e": int(i + 1),
                        "onset": int(i + 1), "peak_z": None, "sign": 1 if y[i + 1] > y[i] else -1})
    for g in gaps:
        i = g["after_index"]
        raw.append({"kind": "sampling_gap", "channel": "__time__", "category": "data_quality", "s": int(i), "e": int(i + 1),
                    "onset": int(i + 1), "peak_z": None, "sign": 0})

    t = table[tcol].to_numpy(float) if tcol else None
    rows = table["__row"].to_numpy()
    unit = tinfo.get("unit_hypothesis")
    scale = unit_to_seconds.get(unit) if tcol else None
    ranges: Dict[str, float] = {}
    floors: Dict[str, float] = {}
    for h in series_ok:
        sc = channel_scale(table[h].to_numpy(float))
        ranges[h] = sc["range"] if sc else 0.0
        floors[h] = sc["floor"] if sc else 0.0

    tol = max(5, w // 4)
    cap = max(50, int(EVENT_MAX_SPAN_FRACTION * n))
    events: List[Dict[str, Any]] = []
    for cat in CATEGORIES:
        items = sorted((r for r in raw if r["category"] == cat), key=lambda r: (r["s"], r["e"]))
        groups: List[List[Dict[str, Any]]] = []
        cur_start = cur_end = -10**9
        for r in items:
            r_end = r.get("e_level") or r["e"]
            if groups and r["s"] <= cur_end + tol and max(cur_end, r_end) - cur_start <= cap:
                groups[-1].append(r)
                cur_end = max(cur_end, r_end)
            else:
                groups.append([r])
                cur_start, cur_end = r["s"], r_end
        for g in groups:
            s, e = min(r["s"] for r in g), max(r.get("e_level") or r["e"] for r in g)
            chs = sorted({r["channel"] for r in g if r["channel"] != "__time__"})
            meas: List[Dict[str, Any]] = []
            for h in chs:
                m = _measure(table[h].to_numpy(float), s, e, [r for r in g if r["channel"] == h], chans[h], h, w, t, rows, ranges.get(h, 0.0), floors.get(h, 0.0))
                if m:
                    meas.append(m)
            meas.sort(key=lambda m: -(min(m["relative_swing"] or 0.0, 1.5) + (0.5 if m["persistence"] and m["persistence"].startswith("sustained") else 0.0)))
            meas = _cap_keep_groups(meas, MAX_EVENT_CHANNELS)
            kinds = sorted({_kind_for_category(r["kind"], cat) for r in g})
            span_raw = r6(t[e] - t[s]) if t is not None else None
            ev = {
                "id": "", "category": cat, "kinds": kinds,
                "channels": [m["channel"] for m in meas] or chs[:MAX_EVENT_CHANNELS],
                "n_channels_involved": len(chs),
                "start_index": int(s), "end_index": int(e), "n_samples": int(e - s + 1),
                "start_row": int(rows[s]), "end_row": int(rows[e]),
                "start_raw": r6(t[s]) if t is not None else None, "end_raw": r6(t[e]) if t is not None else None,
                "span_raw": span_raw,
                "span_s_assumed": ({"value": r6((t[e] - t[s]) * scale, 3),
                                    "basis": "verified" if tinfo.get("status") == "verified" else f"ASSUMING {unit}"}
                                   if (t is not None and scale) else None),
                "measurements": meas, "pad": int(max(3, (e - s) // 2)),
                "_onsets": {h: min(r["onset"] for r in g if r["channel"] == h) for h in chs},
                "_rank": max([m["relative_swing"] or 0.0 for m in meas] or [0.0]) + 0.1 * len(chs) + (0.3 if cat == "statistical_irregularity" else 0.0),
            }
            if cat == "data_quality" and not meas:
                ev["_rank"] = 0.2 + (0.5 if "sampling_gap" in kinds else 0.0)
            events.append(ev)

    overflow = max(0, len(events) - MAX_EVENTS)
    if overflow:
        events = sorted(events, key=lambda x: -x["_rank"])[:MAX_EVENTS]
    events.sort(key=lambda x: (x["start_index"], CATEGORIES.index(x["category"])))
    for i, ev in enumerate(events, 1):
        ev["id"] = f"E{i}"
    _associate(events, table, chans, tol)
    for ev in events:
        for m in ev["measurements"]:
            m["evidence"] = {k: f"{ev['id']}.{m['channel']}.{k}" for k in
                             ("before_median", "during_min", "during_max", "during_median", "after_median", "max_step", "swing",
                              "shift", "shift_sigma", "peak_departure") if m.get(k) is not None
                             and not (k in ("shift_sigma", "peak_departure") and "status_change" in m["kinds"])}
        ev["why_selected"] = why_selected(ev, w)
        ev["priority"] = _priority(ev)
        ev.pop("_rank", None)
        ev.pop("_onsets", None)
    info = {"baseline_window_samples": w, "merge_tolerance_samples": tol, "z_threshold": Z_THRESHOLD, "min_rel_change": MIN_REL_CHANGE,
            "spike_max_samples": SPIKE_MAX_SAMPLES, "variability_ratio": VAR_RATIO, "simultaneity_samples": SIM_TOL,
            "baseline_min_samples": BASELINE_MIN, "post_min_samples": POST_MIN}
    return events, overflow, info


# ---- cross-channel association ---------------------------------------------------------------------------------------

def _associate(events: List[Dict[str, Any]], table: pd.DataFrame, chans: Dict[str, Dict[str, Any]], tol: int) -> None:
    group_sizes: Dict[str, int] = {}
    for h, c in chans.items():
        if h in table and c.get("group") not in (None, "unclassified"):
            group_sizes[c["group"]] = group_sizes.get(c["group"], 0) + 1
    tcol_vals = table["__row"].to_numpy()
    for ev in events:
        by_ch = {m["channel"]: m for m in ev["measurements"]}
        onsets = {h: i for h, i in ev["_onsets"].items() if h in by_ch}
        assoc: List[Dict[str, Any]] = []
        # timing: cluster onsets
        order = sorted(onsets.items(), key=lambda kv: kv[1])
        clusters: List[List[Tuple[str, int]]] = []
        for h, i in order:
            if clusters and i - clusters[-1][-1][1] <= SIM_TOL:
                clusters[-1].append((h, i))
            else:
                clusters.append([(h, i)])
        multi = [c for c in clusters if len(c) >= 2]
        for c in multi:
            assoc.append({"type": "simultaneous", "basis": "observed timing", "channels": [h for h, _ in c],
                          "onset_spread_samples": int(c[-1][1] - c[0][1]), "first_row": int(tcol_vals[c[0][1]]),
                          "physical_relationship": "not established"})
        if len(clusters) >= 2:
            assoc.append({"type": "sequential", "basis": "observed timing", "physical_relationship": "not established",
                          "steps": [{"channels": [h for h, _ in c], "onset_row": int(tcol_vals[c[0][1]]),
                                     "lag_samples": int(c[0][1] - clusters[0][0][1])} for c in clusters]})
        # group co-movement
        bygroup: Dict[str, List[str]] = {}
        for h, m in by_ch.items():
            if m["group"] not in (None, "unclassified", "status_flag") and m["shift"] is not None:
                bygroup.setdefault(m["group"], []).append(h)
        for g, hs in bygroup.items():
            if len(hs) >= 2:
                dirs = {by_ch[h]["direction"] for h in hs}
                assoc.append({"type": "group", "basis": "observed co-movement", "group": g, "channels": hs, "n_in_file": group_sizes.get(g, len(hs)),
                              "direction": dirs.pop() if len(dirs) == 1 else "mixed", "physical_relationship": "not established"})
        # statistical correlation inside the event window
        s, e = ev["start_index"], ev["end_index"]
        lo, hi = max(0, s - ev["pad"]), min(len(table) - 1, e + ev["pad"])
        cand = [h for h in by_ch if not chans[h].get("is_flag")]
        pairs: List[Dict[str, Any]] = []
        if len(cand) >= 2 and hi - lo + 1 >= CORR_MIN_N:
            sub = table[cand].iloc[lo : hi + 1].replace([np.inf, -np.inf], np.nan)
            cm = sub.corr().to_numpy()
            for a in range(len(cand)):
                for b in range(a + 1, len(cand)):
                    if np.isfinite(cm[a, b]) and abs(cm[a, b]) >= CORR_MIN_ABS and sub[[cand[a], cand[b]]].dropna().shape[0] >= CORR_MIN_N:
                        pairs.append({"type": "correlation", "basis": "statistical correlation (this window only)", "a": cand[a], "b": cand[b],
                                      "r": r6(cm[a, b], 3), "n": int(sub[[cand[a], cand[b]]].dropna().shape[0]), "physical_relationship": "not established"})
            pairs.sort(key=lambda p: -abs(p["r"]))
        assoc += pairs[:4]
        ev["associations"] = assoc
    # links to events of other categories that overlap or sit next to this one
    for ev in events:
        links = []
        for o in events:
            if o is ev or o["category"] == ev["category"]:
                continue
            gap = max(ev["start_index"], o["start_index"]) - min(ev["end_index"], o["end_index"])
            if gap <= tol:
                links.append({"event_id": o["id"], "category": o["category"],
                              "relation": "overlapping" if gap <= 0 else f"within {gap} samples"})
        ev["concurrent_with"] = links


# ---- wording of why an interval was chosen (deterministic) ---------------------------------------------------------

_KIND_WHY = {
    "spike": "a brief departure of at most {sm} samples that is far outside the channel's recent spread",
    "sustained_shift": "a departure from the channel's own preceding median that was still present afterwards (or until the file ended)",
    "transient_excursion": "a departure from the channel's preceding median that lasted more than {sm} samples and then returned",
    "variability_increase": "a block of samples whose spread is much larger than the preceding blocks",
    "flatline": "a run of exactly repeated values in a channel that otherwise varies",
    "missing_values": "a run of missing or non-finite values",
    "sampling_gap": "a step between timestamps much larger than the median step",
    "status_change": "a status/flag channel changed value",
    "command_change": "a pilot-command channel moved away from its preceding median",
}


def why_selected(ev: Dict[str, Any], w: int) -> str:
    kinds = [k for k in ev["kinds"]]
    parts = [_KIND_WHY[k].format(sm=SPIKE_MAX_SAMPLES) for k in kinds if k in _KIND_WHY]
    base = ("Selected for: " + "; ".join(parts) + ".") if parts else "Selected by the statistical screen."
    if ev["category"] in ("statistical_irregularity", "state_transition") and any(k not in ("status_change", "flatline", "missing_values", "sampling_gap") for k in kinds):
        base += (f" A change counts when it exceeds {Z_THRESHOLD:g} robust sigmas of the preceding {w} samples and at least "
                 f"{int(MIN_REL_CHANGE * 100)}% of the channel's own range.")
    return base


def _priority(ev: Dict[str, Any]) -> str:
    """'review_first' | 'review' | 'minor' -- ordering aid only; it is not a severity or a probability."""
    cat = ev["category"]
    n = ev.get("n_channels_involved", 0)
    sustained = any((m.get("persistence") or "").startswith("sustained") for m in ev["measurements"])
    multi_assoc = any(a["type"] in ("simultaneous", "group") for a in ev.get("associations", []))
    linked = bool(ev.get("concurrent_with"))
    if cat == "statistical_irregularity" and (n >= 3 or (sustained and multi_assoc)):
        return "review_first"
    if cat == "data_quality" and ("sampling_gap" in ev["kinds"] or linked):
        return "review"
    if cat == "statistical_irregularity":
        return "review"
    if cat == "state_transition":
        return "review" if linked and n >= 3 else "minor"
    return "minor"
