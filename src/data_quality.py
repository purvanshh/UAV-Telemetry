"""Structured data-quality / coverage accounting for one analysis.

Every analysis reports how much of the input was actually scored. Counts are
defined once, here, and each unit is counted exactly once:

* **row**        – one line of an uploaded CSV. ``rows_input`` = every line;
  ``rows_missing_timestamp`` + ``rows_duplicate_timestamp`` are dropped
  (disjoint: a duplicate is only counted among rows with a valid timestamp);
  ``rows_after_cleaning`` = ``rows_input - missing - duplicate``.
* **grid sample** – one tick of the 10 Hz alignment grid over the interval in
  which IMU, ATT and BARO all have data. Grid samples inside a source gap
  (``_session_id < 0``) are *excluded*; the others belong to exactly one session.
* **candidate window** – a 30-sample window at stride 15 that fits entirely
  inside one session. Windows cannot exist in gaps or in sessions shorter than
  a window, so those samples are reported separately, not as discarded windows.
* **scored window** – a candidate window whose 12 deployed features are all
  finite. ``discarded = candidate - scored`` (each window once).
* **analysed duration** – length of the union of scored windows' time spans.

``MIN_*`` constants decide ``coverage_status``; they only add warnings and a
status — they never alter which windows are scored.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

SCHEMA = "1.0"
MIN_ANALYSED_FRACTION = 0.5  # analysed / aligned duration below this => low_coverage
MAX_DISCARD_FRACTION = 0.5  # discarded / candidate windows above this => low_coverage
MAX_LISTED_INTERVALS = 100

DEFINITIONS = {
    "rows_input": "Lines in the uploaded CSV (data rows).",
    "rows_missing_timestamp": "Rows with an empty/non-numeric timestamp; dropped.",
    "rows_duplicate_timestamp": "Rows (valid timestamp) repeating an earlier timestamp; dropped, first kept.",
    "rows_after_cleaning": "rows_input - rows_missing_timestamp - rows_duplicate_timestamp.",
    "grid_samples": "10 Hz ticks over the IMU/ATT/BARO overlap interval.",
    "grid_samples_in_gaps": "Grid samples inside a source gap (session_id < 0); excluded from every window.",
    "candidate_windows": "30-sample windows at stride 15 that fit inside one session.",
    "scored_windows": "Candidate windows with all deployed features finite.",
    "discarded_windows": "candidate_windows - scored_windows (each counted once).",
    "analysed_duration_s": "Union length of scored windows' time spans (t_start .. t_end + one sample).",
}


def _runs(mask: np.ndarray, t: np.ndarray, step: float) -> List[Dict[str, float]]:
    """Contiguous True runs of ``mask`` as ``[start_s, end_s]`` intervals."""
    out: List[Dict[str, float]] = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append({"start_s": float(t[i]), "end_s": float(t[j] + step)})
            i = j + 1
        else:
            i += 1
    return out


def _union_length(intervals: Sequence[Sequence[float]]) -> float:
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return float(total)


def account_windows(
    aligned: pd.DataFrame,
    features: Sequence[str],
    win_len: int,
    stride: int,
    step_s: float,
) -> Dict[str, Any]:
    """Mirror ``make_windows`` window enumeration to count candidates and reasons.

    This re-derives the candidate set from ``_session_id`` and the feature matrix;
    callers compare ``scored`` against the real batch size (see ``consistent``).
    """
    t = aligned["_t_s"].to_numpy()
    sid = aligned["_session_id"].to_numpy()
    X = aligned[list(features)].to_numpy(dtype=np.float64)
    finite = np.isfinite(X)
    candidate = scored = 0
    by_feature = {f: 0 for f in features}
    bad_intervals: List[List[float]] = []
    good_intervals: List[List[float]] = []
    short_sessions: List[Dict[str, float]] = []
    short_samples = 0
    for s in np.unique(sid):
        if s < 0:
            continue
        idx = np.where(sid == s)[0]
        if len(idx) < win_len:
            short_samples += len(idx)
            short_sessions.append(
                {"start_s": float(t[idx[0]]), "end_s": float(t[idx[-1]] + step_s)}
            )
            continue
        for s0 in range(0, len(idx) - win_len + 1, stride):
            w = idx[s0 : s0 + win_len]
            candidate += 1
            iv = [float(t[w[0]]), float(t[w[-1]] + step_s)]
            ok = finite[w].all(axis=0)
            if ok.all():
                scored += 1
                good_intervals.append(iv)
            else:
                bad_intervals.append(iv)
                for j in np.where(~ok)[0]:
                    by_feature[features[j]] += 1
    return {
        "candidate": candidate,
        "scored": scored,
        "by_feature": {k: v for k, v in by_feature.items() if v},
        "bad_intervals": bad_intervals,
        "good_intervals": good_intervals,
        "short_sessions": short_sessions,
        "short_session_samples": short_samples,
    }


def _stream_block(
    stream: str,
    raw: pd.DataFrame,
    tb: Any,
    clean: Optional[pd.DataFrame],
    used_columns: Sequence[str],
    filename: Optional[str],
    file_hash: Optional[str],
) -> Dict[str, Any]:
    cols = {str(c).lower().replace("_", "").replace(" ", ""): c for c in raw.columns}
    invalid: Dict[str, int] = {}
    for c in used_columns:
        src = cols.get(c.lower())
        if src is None:
            invalid[c] = int(len(raw))  # column absent: every row unusable
            continue
        v = pd.to_numeric(raw[src], errors="coerce").to_numpy(dtype="float64")
        invalid[c] = int((~np.isfinite(v)).sum())
    gaps = tb.gaps
    return {
        "filename": filename,
        "sha256": file_hash,
        "rows_input": tb.n_rows,
        "rows_missing_timestamp": tb.n_missing,
        "rows_duplicate_timestamp": tb.n_duplicates,
        "rows_after_cleaning": tb.n_unique,
        "non_monotonic_steps_in_file_order": tb.n_nonmonotonic_steps,
        "timestamp_column": tb.column,
        "timestamp_unit": tb.unit,
        "timestamp_unit_source": tb.unit_source,
        "raw_timestamp_first": tb.raw_first,
        "raw_timestamp_last": tb.raw_last,
        "t_start_s": tb.t_start_s,
        "t_end_s": tb.t_end_s,
        "duration_s": tb.duration_s,
        "median_dt_s": tb.median_dt_s,
        "est_hz": tb.est_hz,
        "gap_count": len(gaps),
        "gap_total_s": float(sum(g["duration_s"] for g in gaps)),
        "gap_max_s": tb.max_gap_s if gaps else None,
        "gaps": gaps[:MAX_LISTED_INTERVALS],
        "invalid_values_by_used_column": invalid,
        "warnings": list(tb.warnings),
    }


def build_data_quality(
    *,
    frames: Dict[str, pd.DataFrame],
    detail: Any,  # AlignmentDetail
    deployed_features: Sequence[str],
    candidate_features: Sequence[str],
    used_source_columns: Dict[str, Sequence[str]],
    win_samples: int,
    stride_samples: int,
    n_scored: Optional[int] = None,
    filenames: Optional[Dict[str, str]] = None,
    file_hashes: Optional[Dict[str, str]] = None,
    upstream_warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Assemble the data-quality summary (JSON-safe dict)."""
    filenames = filenames or {}
    file_hashes = file_hashes or {}
    aligned: pd.DataFrame = detail.aligned
    cfg = detail.cfg
    hz = float(cfg["target_hz"])
    step = 1.0 / hz
    warnings: List[str] = []

    streams = {
        s: _stream_block(
            s,
            frames[s],
            detail.timebase[s],
            detail.clean_streams.get(s),
            used_source_columns.get(s, ()),
            filenames.get(s),
            file_hashes.get(s),
        )
        for s in ("IMU", "ATT", "BARO")
    }

    # ---- timeline / durations -------------------------------------------
    starts = [b["t_start_s"] for b in streams.values() if b["t_start_s"] is not None]
    ends = [b["t_end_s"] for b in streams.values() if b["t_end_s"] is not None]
    flight_start, flight_end = (min(starts), max(ends)) if starts else (None, None)
    flight_duration = (flight_end - flight_start) if starts else None

    t = aligned["_t_s"].to_numpy()
    sid = aligned["_session_id"].to_numpy()
    n_grid = int(len(aligned))
    in_gap = sid < 0
    sessions = [int(x) for x in np.unique(sid) if x >= 0]
    aligned_duration = n_grid * step

    # ---- per-feature availability / resampling ---------------------------
    per_feature: Dict[str, Any] = {}
    for f in candidate_features:
        if f not in aligned.columns:
            per_feature[f] = {"status": "missing", "finite_samples": 0, "nan_samples": n_grid}
            continue
        fin = np.isfinite(aligned[f].to_numpy(dtype=float))
        per_feature[f] = {
            "status": "available" if fin.any() else "missing",
            "finite_samples": int(fin.sum()),
            "nan_samples": int((~fin).sum()),
            "nan_fraction": float((~fin).mean()) if n_grid else None,
        }
    resampling = {}
    for s, clean in detail.clean_streams.items():
        src_t = clean["__t_s"].to_numpy()
        exact = int(np.isin(t, src_t).sum())
        resampling[s] = {
            "method": "gap-aware linear interpolation onto the 10 Hz grid",
            "source_samples": int(len(src_t)),
            "grid_samples_exactly_on_source_sample": exact,
            "grid_samples_interpolated_or_gap": n_grid - exact,
        }

    # ---- windows ----------------------------------------------------------
    acc = account_windows(aligned, deployed_features, win_samples, stride_samples, step)
    candidate, scored = acc["candidate"], acc["scored"]
    discarded = candidate - scored
    consistent = n_scored is None or n_scored == scored
    if not consistent:
        warnings.append(
            f"Window accounting ({scored}) disagrees with the scored batch "
            f"({n_scored}); counts below may be unreliable."
        )
    analysed = _union_length(acc["good_intervals"])
    gap_runs = _runs(in_gap, t, step)
    excluded = (
        [{**r, "reason": "source_gap"} for r in gap_runs]
        + [{**r, "reason": "session_shorter_than_window"} for r in acc["short_sessions"]]
    )
    bad_runs = [
        {"start_s": s, "end_s": e, "reason": "non_finite_feature_values"}
        for s, e in _merge(acc["bad_intervals"])
    ]
    excluded += bad_runs

    # ---- features ---------------------------------------------------------
    deployed = list(deployed_features)
    avail = [f for f in deployed if per_feature.get(f, {}).get("status") == "available"]
    missing = [f for f in deployed if f not in avail]
    excluded_feats = [f for f in candidate_features if f not in deployed]

    # ---- status & warnings ------------------------------------------------
    cov_aligned = (analysed / aligned_duration) if aligned_duration > 0 else 0.0
    discard_frac = (discarded / candidate) if candidate else None
    if scored == 0:
        status = "insufficient_data"
        warnings.append(
            "No windows could be scored: this analysis did not examine any of the flight."
        )
    elif cov_aligned < MIN_ANALYSED_FRACTION or (
        discard_frac is not None and discard_frac > MAX_DISCARD_FRACTION
    ):
        status = "low_coverage"
        warnings.append(
            f"Low coverage: only {cov_aligned:.0%} of the aligned flight "
            f"({analysed:.1f} s of {aligned_duration:.1f} s) was scored"
            + (f"; {discard_frac:.0%} of candidate windows were discarded." if discard_frac is not None else ".")
            + " A result with no flags does not describe the unscored portion."
        )
    else:
        status = "ok"
    if missing:
        warnings.append(f"Deployed feature(s) unavailable in this flight: {', '.join(missing)}.")
    if gap_runs:
        warnings.append(
            f"{len(gap_runs)} source-gap interval(s) totalling "
            f"{sum(r['end_s'] - r['start_s'] for r in gap_runs):.1f} s were excluded."
        )
    warnings.extend(upstream_warnings or [])
    for b in streams.values():
        warnings.extend(w for w in b["warnings"] if w not in warnings)

    return {
        "schema": SCHEMA,
        "coverage_status": status,
        "definitions": DEFINITIONS,
        "streams": streams,
        "timeline": {
            "time_reference": "source log clock, seconds (TimeUS / 1e6); not wall-clock time",
            "source_start_s": flight_start,
            "source_end_s": flight_end,
            "aligned_start_s": float(t[0]),
            "aligned_end_s": float(t[-1] + step),
        },
        "durations": {
            "flight_duration_s": flight_duration,
            "aligned_duration_s": aligned_duration,
            "analysed_duration_s": analysed,
            "analysed_fraction_of_aligned": cov_aligned,
            "analysed_fraction_of_flight": (analysed / flight_duration) if flight_duration else None,
        },
        "alignment": {
            "target_hz": hz,
            "max_gap_s": float(cfg["max_gap_s"]),
            "session_gap_s": float(cfg["session_gap_s"]),
            "grid_samples": n_grid,
            "grid_samples_in_gaps": int(in_gap.sum()),
            "grid_samples_in_sessions": int((~in_gap).sum()),
            "session_count": len(sessions),
            "resampling": resampling,
            "per_feature": per_feature,
        },
        "windows": {
            "window_samples": win_samples,
            "stride_samples": stride_samples,
            "candidate": candidate,
            "scored": scored,
            "discarded": discarded,
            "discarded_pct": (100.0 * discarded / candidate) if candidate else None,
            "discard_reasons": {
                "non_finite_feature_values": discarded,
                "non_finite_by_feature_not_exclusive": acc["by_feature"],
            },
            "samples_in_sessions_shorter_than_window": acc["short_session_samples"],
            "consistent_with_scored_batch": consistent,
        },
        "excluded_intervals": excluded[:MAX_LISTED_INTERVALS],
        "excluded_interval_count": len(excluded),
        "features": {
            "deployed": deployed,
            "available": avail,
            "missing": missing,
            "excluded_from_model": excluded_feats,
        },
        "warnings": warnings,
    }


def _merge(intervals: Sequence[Sequence[float]]) -> List[List[float]]:
    out: List[List[float]] = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([float(s), float(e)])
    return out


def minimal_data_quality(reason: str, status: str = "insufficient_data") -> Dict[str, Any]:
    """Data-quality stub for analyses that failed before alignment."""
    return {
        "schema": SCHEMA,
        "coverage_status": status,
        "definitions": DEFINITIONS,
        "warnings": [reason],
    }
