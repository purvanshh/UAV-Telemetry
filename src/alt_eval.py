"""Phase 3/4 evaluation: alternative detectors vs the aggregate LSTM baseline.

Protocol (fixed before looking at results; encoded below and written to the report)
----------------------------------------------------------------------------------
1. **Burden set** = unreviewed real flights in the ``val`` split. Reference
   statistics are fitted there, and each alternative's threshold is chosen so its
   flagged-window fraction on the burden set equals the baseline's at the deployed
   threshold (*matched alert burden*). Nothing is fitted on ``test`` flights.
2. **Held-out evaluation** = ``test`` split sources. Burden is re-measured there
   (flagged-window fraction and alert events per analysed hour — unreviewed, so a
   *burden*, not a false-alarm rate) and the same seeded synthetic injections as
   the Phase 2 evaluation are scored by every detector.
3. **Screen** (synthetic only). An alternative passes iff, on held-out sources:
   (a) test alert burden ≤ ``MAX_BURDEN_RATIO`` × baseline, both as events per
       analysed hour and as flagged-window fraction;
   (b) net gain ≥ ``MIN_NET_GAIN`` injections (detected by it and not by the
       baseline, minus the reverse), on injections evaluable for both;
   (c) net gain ≥ 0 in every held-out flight and every flight-phase stratum with
       ≥ ``MIN_STRATUM_N`` paired injections (consistency across conditions).
   Passing the screen is **not** evidence of real-world benefit: no reviewed
   labels exist, so false alarms cannot be measured. When reviewed intervals are
   supplied, event recall and false alarms per flight-hour are added per detector.
4. Detectors needing the attitude/gyro relation are excluded unless the Euler→
   body-rate convention verifies on every real flight (``verify_rate_convention``).

The LSTM and its threshold are never modified; window length varies only for the
side detectors (the LSTM input is fixed at 30 samples).
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from src.alt_detectors import (
    DETECTORS, DetectorDef, Reference, WindowSet, build_windowset, fit_reference, match_threshold,
)
from src.artifact_loader import load_artifacts
from src.context_features import derive_signals, verify_rate_convention
from src.data_quality import _union_length
from src.detector import UAVDetector
from src.eval_corpus import resolve_flight
from src.event_builder import DEFAULT_MAX_GAP_S, build_events
from src.preprocessing_adapter import align_uploaded_flight_detailed
from src.reviewed_eval import evaluate as evaluate_reviewed, validate_intervals
from src.split_manifest import validate_manifest
from src.synthetic_eval import (
    DEFAULT_LEVELS, INJECTION_TYPES, apply_injection, case_seed, detection_at_threshold,
    plan_injection, sigma_by_feature,
)

SCHEMA = "alt_detector_evaluation/1"
MAX_BURDEN_RATIO = 1.25
MIN_NET_GAIN = 3
MIN_STRATUM_N = 5
REPS = 2


def _prepare(frames, art, det):
    detail = align_uploaded_flight_detailed(dict(frames), art.model_dir)
    cfg = detail.cfg
    sig = derive_signals(detail.aligned, detail.clean_streams["ATT"], detail.prep.gap_aware_interp, float(cfg["max_gap_s"]))
    mu = np.asarray(art.scaler["mean"], dtype=np.float32)
    sd = np.asarray(art.scaler["std"], dtype=np.float32)
    sets = {
        30: build_windowset(detail.aligned, sig, art.features, mu, sd, 30, 15, det),
        10: build_windowset(detail.aligned, sig, art.features, mu, sd, 10, 5, None),
    }
    return detail, sig, sets


def _wdf(ws: WindowSet, scores: np.ndarray, tau: float) -> pd.DataFrame:
    d = ws.meta.copy()
    d["anomaly_score"] = scores
    d["is_anomaly"] = np.where(np.isfinite(scores), scores > tau, False)
    return d


def _burden(ws: WindowSet, scores: np.ndarray, tau: float) -> Dict[str, Any]:
    fin = np.isfinite(scores)
    flagged = fin & (scores > tau)
    events = build_events(_wdf(ws, scores, tau), max_gap_s=DEFAULT_MAX_GAP_S)
    ivs = [[float(a), float(b) + 0.1] for a, b in zip(ws.meta["t_start"][fin], ws.meta["t_end"][fin])]
    hours = _union_length(ivs) / 3600.0
    return {
        "windows_scored": int(fin.sum()), "windows_flagged": int(flagged.sum()),
        "flagged_fraction": float(flagged.sum() / fin.sum()) if fin.any() else None,
        "events": len(events), "analysed_hours": hours,
        "events_per_hour": (len(events) / hours) if hours > 0 else None,
    }


def _inj_record(rec: Mapping[str, Any], base_ws, base_s, inj_ws, inj_s) -> Dict[str, Any]:
    t0, t1 = rec["injected_raw_interval_s"]

    def ov(ws, s):
        m = (ws.meta["t_start"].to_numpy() <= t1) & (ws.meta["t_end"].to_numpy() >= t0) & np.isfinite(s)
        return m

    mi, mb = ov(inj_ws, inj_s), ov(base_ws, base_s)
    return {
        "injection_id": rec["injection_id"], "type": rec["type"],
        "eligible_for_scoring": bool(mi.any()),
        "baseline_max_overlap_score": float(base_s[mb].max()) if mb.any() else None,
        "max_overlap_score": float(inj_s[mi].max()) if mi.any() else None,
        "t_start": t0, "pattern_preserved": rec.get("pattern_preserved", True),
        "overlap_windows": [
            {"t_start": float(a), "t_end": float(b), "score": float(c)}
            for a, b, c in zip(inj_ws.meta["t_start"][mi], inj_ws.meta["t_end"][mi], inj_s[mi])
        ],
    }


def _status(r: Mapping[str, Any], tau: float) -> str:
    if not r["eligible_for_scoring"]:
        return "not_eligible"
    if (r["baseline_max_overlap_score"] or -np.inf) > tau:
        return "confounded"
    return "detected" if (r["max_overlap_score"] or -np.inf) > tau else "missed"


def _paired(rs_d, tau_d, rs_b, tau_b) -> Dict[str, Any]:
    b = {r["injection_id"]: _status(r, tau_b) for r in rs_b}
    gain, loss, both, neither = [], [], 0, 0
    for r in rs_d:
        sd, sb = _status(r, tau_d), b.get(r["injection_id"])
        if sd in ("detected", "missed") and sb in ("detected", "missed"):
            if sd == "detected" and sb == "missed":
                gain.append(r["injection_id"])
            elif sd == "missed" and sb == "detected":
                loss.append(r["injection_id"])
            elif sd == "detected":
                both += 1
            else:
                neither += 1
    return {"n_paired": len(gain) + len(loss) + both + neither, "gained": gain, "lost": loss,
            "both_detected": both, "neither_detected": neither, "net_gain": len(gain) - len(loss)}


def run_alt_evaluation(
    manifest: Mapping[str, Any],
    *,
    seed: int,
    reviewed: Optional[Mapping[str, Any]] = None,
    model_dir=None,
    levels: Mapping[str, List[Dict[str, Any]]] = DEFAULT_LEVELS,
    reps: int = REPS,
    timestamp: Optional[str] = None,
    art=None,
) -> Dict[str, Any]:
    art = art or load_artifacts(model_dir)
    det = UAVDetector(art)
    sigma = sigma_by_feature(art)
    tau0 = art.threshold
    mv = validate_manifest(manifest)
    if not mv["ok"]:
        raise ValueError(f"split manifest invalid: {mv['errors']}")
    usable = [f for f in manifest["flights"] if f.get("split") in ("val", "test")
              and f.get("data_available") and f.get("duplicate_of") is None]
    sources = [f for f in usable if f["eval_role"] == "synthetic_source"]
    burden_flights = [f for f in sources if f["split"] == "val" and f["kind"] == "real_unreviewed"]
    test_flights = [f for f in sources if f["split"] == "test"]

    base: Dict[str, Any] = {}
    frames_by: Dict[str, Any] = {}
    for f in sources:
        fr = resolve_flight(f)
        frames_by[f["flight_id"]] = fr
        base[f["flight_id"]] = _prepare(fr, art, det)

    out: Dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": timestamp or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": {
            "burden_set": [f["flight_id"] for f in burden_flights],
            "held_out": [f["flight_id"] for f in test_flights],
            "max_burden_ratio": MAX_BURDEN_RATIO, "min_net_gain": MIN_NET_GAIN, "min_stratum_n": MIN_STRATUM_N,
            "doc": __doc__,
        },
        "baseline_threshold": tau0, "seed": seed,
        "limitations": [],
    }
    if not burden_flights or not test_flights:
        out["status"] = "insufficient_corpus"
        out["reason"] = "Need >=1 unreviewed real flight in val (burden set) and >=1 source in test."
        return out

    # ---- label-free convention check on all real, un-perturbed flights ----
    real = [f["flight_id"] for f in sources if f["kind"] == "real_unreviewed"]
    verif = verify_rate_convention(
        {fid: base[fid][1] for fid in base}, {fid: base[fid][0].aligned for fid in base}, gating=real)
    out["rate_convention"] = verif

    # ---- reference + matched thresholds (burden set only) ----
    ref = fit_reference({w: [base[f["flight_id"]][2][w] for f in burden_flights] for w in (30, 10)})
    bws = [base[f["flight_id"]][2][30] for f in burden_flights]
    n_val = sum(w.n for w in bws)
    flagged0 = sum(int((w.lstm > tau0).sum()) for w in bws)
    target = flagged0 / n_val
    out["burden_set"] = {"windows": n_val, "baseline_flagged": flagged0, "target_flagged_fraction": target,
                         "reference_windows": ref.n_windows,
                         "phase_windows": dict(Counter(np.concatenate([w.phase for w in bws]).tolist()))}
    phase_counts = out["burden_set"]["phase_windows"]
    from src.alt_detectors import MIN_PHASE_WINDOWS
    phase_ok = sum(1 for v in phase_counts.values() if v >= MIN_PHASE_WINDOWS) >= 2
    out["burden_set"]["phase_context_usable"] = phase_ok
    dets: List[DetectorDef] = []
    excluded: Dict[str, str] = {}
    for d in DETECTORS:
        if d.needs_rate_convention and not verif["verified"]:
            excluded[d.name] = "attitude/gyro convention not verified on every real flight"
            continue
        dets.append(d)
    taus: Dict[str, float] = {}
    for d in dets:
        sc = np.concatenate([d.score(base[f["flight_id"]][2][d.win], ref) for f in burden_flights])
        taus[d.name] = tau0 if d.name == "lstm_aggregate" else match_threshold(sc, target)
    out["excluded_detectors"] = excluded
    out["thresholds"] = {d.name: taus[d.name] for d in dets}

    # ---- baseline-flight scores per detector ----
    bscore = {d.name: {fid: d.score(base[fid][2][d.win], ref) for fid in base} for d in dets}

    def burden_block(fids):
        res = {}
        for d in dets:
            per = {fid: _burden(base[fid][2][d.win], bscore[d.name][fid], taus[d.name]) for fid in fids}
            ev = sum(p["events"] for p in per.values())
            hr = sum(p["analysed_hours"] for p in per.values())
            fl = sum(p["windows_flagged"] for p in per.values())
            sc = sum(p["windows_scored"] for p in per.values())
            res[d.name] = {"per_flight": per, "events": ev, "analysed_hours": hr,
                           "events_per_hour": (ev / hr) if hr else None,
                           "flagged_fraction": (fl / sc) if sc else None}
        return res

    burden_val = burden_block([f["flight_id"] for f in burden_flights])
    burden_test = burden_block([f["flight_id"] for f in test_flights])

    # ---- synthetic injections (same seeds/placement as Phase 2) ----
    recs: Dict[str, List[Dict[str, Any]]] = {d.name: [] for d in dets}
    for f in sources:
        fid, kind = f["flight_id"], f["kind"]
        detail, sig, sets = base[fid]
        w30 = sets[30]
        bw = w30.meta.copy()
        bw["anomaly_score"] = w30.lstm
        if len(bw) == 0:
            continue
        for itype in INJECTION_TYPES:
            for li, lvl in enumerate(levels[itype]):
                for rep in range(reps):
                    iid = f"{fid}:{itype}:L{li}:r{rep}"
                    cs = case_seed(seed, fid, f["content_hash"], itype, li, rep)
                    spec = plan_injection(bw, itype, lvl, np.random.default_rng(cs), cs, iid, tau0)
                    if spec is None:
                        continue
                    new_frames, rec = apply_injection(frames_by[fid], spec, sigma)
                    t0, t1 = rec["injected_raw_interval_s"]
                    ov = (w30.meta["t_start"] <= t1) & (w30.meta["t_end"] >= t0)
                    ph = Counter(w30.phase[ov.to_numpy()].tolist()).most_common(1)
                    stratum = ph[0][0] if ph else "unknown"
                    _d, _s, isets = _prepare(new_frames, art, det)
                    for d in dets:
                        r = _inj_record({**rec, "injection_id": iid, "type": itype},
                                        sets[d.win], bscore[d.name][fid], isets[d.win], d.score(isets[d.win], ref))
                        r.update(split=f["split"], flight=fid, phase=stratum)
                        recs[d.name].append(r)

    # ---- summaries ----
    results: Dict[str, Any] = {}
    b = "lstm_aggregate"
    for d in dets:
        tau = taus[d.name]
        test_r = [r for r in recs[d.name] if r["split"] == "test"]
        val_r = [r for r in recs[d.name] if r["split"] == "val"]
        test_b = [r for r in recs[b] if r["split"] == "test"]
        ent: Dict[str, Any] = {
            "family": d.family, "window_samples": d.win, "description": d.description, "threshold": tau,
            "burden_val_in_sample": burden_val[d.name],
            "burden_test": burden_test[d.name],
            "synthetic_test": {
                "overall": detection_at_threshold(test_r, tau),
                "by_type": {t: detection_at_threshold([r for r in test_r if r["type"] == t], tau)
                            for t in INJECTION_TYPES if any(r["type"] == t for r in test_r)},
                "by_flight": {fid: detection_at_threshold([r for r in test_r if r["flight"] == fid], tau)
                              for fid in sorted({r["flight"] for r in test_r})},
                "by_phase": {p: detection_at_threshold([r for r in test_r if r["phase"] == p], tau)
                             for p in sorted({r["phase"] for r in test_r})},
            },
            "synthetic_val_in_sample": detection_at_threshold(val_r, tau),
        }
        if d.name != b:
            tb = burden_test[b]["events_per_hour"]
            td = burden_test[d.name]["events_per_hour"]
            ratio = (td / tb) if (tb and td is not None) else None
            ent["burden_ratio_test_events_per_hour"] = ratio
            ent["burden_ratio_test_flagged_fraction"] = (
                burden_test[d.name]["flagged_fraction"] / burden_test[b]["flagged_fraction"]
                if burden_test[b]["flagged_fraction"] else None)
            overall = _paired(test_r, tau, test_b, tau0)
            strata = {}
            for key, vals in (("flight", sorted({r["flight"] for r in test_r})),
                              ("phase", sorted({r["phase"] for r in test_r}))):
                for v in vals:
                    sd = [r for r in test_r if r[key] == v]
                    sb = [r for r in test_b if r[key] == v]
                    p = _paired(sd, tau, sb, tau0)
                    strata[f"{key}:{v}"] = {"n_paired": p["n_paired"], "gained": len(p["gained"]),
                                            "lost": len(p["lost"]), "net_gain": p["net_gain"]}
            fails = []
            if ratio is None or ratio > MAX_BURDEN_RATIO:
                fails.append(f"events/hour ratio {ratio if ratio is None else round(ratio, 2)} > {MAX_BURDEN_RATIO}")
            fr = ent["burden_ratio_test_flagged_fraction"]
            if fr is None or fr > MAX_BURDEN_RATIO:
                fails.append(f"flagged-window ratio {fr if fr is None else round(fr, 2)} > {MAX_BURDEN_RATIO}")
            if d.name.endswith("_phase") and not phase_ok:
                fails.append("phase context degenerate on the burden set (fewer than 2 phases with enough windows)")
            if overall["net_gain"] < MIN_NET_GAIN:
                fails.append(f"net gain {overall['net_gain']} < {MIN_NET_GAIN}")
            for k, s in strata.items():
                if s["n_paired"] >= MIN_STRATUM_N and s["net_gain"] < 0:
                    fails.append(f"negative net gain in {k} ({s['net_gain']})")
            ent["paired_vs_baseline_test"] = {**overall, "strata": strata}
            ent["screen"] = {"passed": not fails, "failures": fails,
                             "scope": "SYNTHETIC screen only; real-world benefit not demonstrated."}
        if reviewed and reviewed.get("intervals"):
            rv = validate_intervals(reviewed)
            wins = {f["flight_id"]: _wdf(base[f["flight_id"]][2][d.win], bscore[d.name][f["flight_id"]], tau)
                    for f in sources if f["kind"] == "real_unreviewed"}
            r = evaluate_reviewed(wins, rv, threshold=tau)
            ent["reviewed"] = {k: r.get(k) for k in ("status", "event_recall", "false_alarms_per_flight_hour",
                                                    "false_alarm_events", "reviewed_unusual_intervals",
                                                    "expected_hours_reviewed")}
        results[d.name] = ent
    out["detectors"] = results
    out["status"] = "ok"
    passed = [n for n, e in results.items() if e.get("screen", {}).get("passed")]
    out["screen_summary"] = {"passed": passed, "n_alternatives": len(results) - 1,
                             "note": "Many alternatives were screened on a handful of injections; passes are "
                                     "hypotheses to confirm on reviewed real labels, not findings."}
    out["real_world"] = (
        {"status": "reviewed_labels_evaluated"} if reviewed and reviewed.get("intervals") else
        {"status": "no_reviewed_labels",
         "reason": "Additional LABELLED real anomalies cannot be counted and false alarms cannot be measured. "
                   "The Phase 3/4 success criteria are NOT demonstrated on real data."})
    out["limitations"] = [
        f"Burden set is {len(burden_flights)} unreviewed real flight(s) with {n_val} windows; matched thresholds are coarse.",
        f"Held-out sources: {', '.join(f['flight_id'] for f in test_flights)} "
        f"({sum(1 for f in test_flights if f['kind']=='real_unreviewed')} real).",
        "Alert burden is unreviewed events per analysed hour — not a false-alarm rate.",
        "Labelled anomalies are synthetic injections generated by this repository.",
        "LSTM window length is fixed at 30 samples; revised window settings apply only to side detectors.",
        "No retraining was done; context signals feed side detectors, not the deployed model.",
    ]
    return out


def render_alt_markdown(r: Mapping[str, Any]) -> str:
    L = ["# Alternative detectors vs the aggregate baseline (Phases 3–4)\n"]
    L.append(f"_Generated {r['generated_at']}_\n")
    if r["status"] != "ok":
        L.append(f"**Not run:** {r['reason']}\n")
        return "\n".join(L)
    p = r["protocol"]
    L.append(f"Burden set: {p['burden_set']} · held-out: {p['held_out']} · rule: burden ≤ {p['max_burden_ratio']}× baseline, "
             f"net gain ≥ {p['min_net_gain']}, no negative stratum (n ≥ {p['min_stratum_n']}).\n")
    bs = r["burden_set"]
    L.append(f"Matched alert burden: baseline flags {bs['baseline_flagged']}/{bs['windows']} burden-set windows "
             f"({100 * bs['target_flagged_fraction']:.2f}%); each alternative's threshold is matched to that.\n")
    rc = r["rate_convention"]
    L.append(f"Euler→body-rate convention verified on real flights: **{rc['verified']}**. "
             + ("" if rc["verified"] else f"({rc['reason']}) "))
    for row in rc["rows"]:
        if row["gating"] and row["zyx"]:
            L.append(f"  - {row['flight_id']} {row['axis']}: ZYX corr {row['zyx']['corr']:.2f} slope {row['zyx']['slope']:.2f}"
                     + (f"; naive corr {row['naive']['corr']:.2f}" if row["naive"] else ""))
    if r["excluded_detectors"]:
        L.append(f"\nExcluded: {r['excluded_detectors']}")
    L.append("\n## Held-out results (synthetic injections; burden unreviewed)\n")
    L.append("| detector | win | threshold | test events/h | burden × | detected / evaluable | gained | lost | net | screen |")
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|---|")
    for n, e in r["detectors"].items():
        o = e["synthetic_test"]["overall"]
        pv = e.get("paired_vs_baseline_test")
        sc = e.get("screen")
        eph = e["burden_test"]["events_per_hour"]
        L.append(f"| {n} | {e['window_samples']} | {e['threshold']:.4g} | {'n/a' if eph is None else f'{eph:.1f}'} | "
                 f"{'—' if e.get('burden_ratio_test_events_per_hour') is None else format(e['burden_ratio_test_events_per_hour'], '.2f')} | "
                 f"{o['n_detected']} / {o['n_evaluable']} | {len(pv['gained']) if pv else '—'} | {len(pv['lost']) if pv else '—'} | "
                 f"{pv['net_gain'] if pv else '—'} | {('PASS' if sc['passed'] else 'fail') if sc else 'baseline'} |")
    L.append("\nFailures:")
    for n, e in r["detectors"].items():
        if e.get("screen") and not e["screen"]["passed"]:
            L.append(f"- `{n}`: " + "; ".join(e["screen"]["failures"]))
    L.append(f"\nScreen passed: {r['screen_summary']['passed'] or 'none'}. {r['screen_summary']['note']}\n")
    L.append("## Real-world\n")
    L.append(r["real_world"].get("reason", "Reviewed labels evaluated; see JSON.") + "\n")
    L.append("## Limitations\n")
    L.extend(f"- {x}" for x in r["limitations"])
    L.append("\n## Reproduce\n```bash\npython scripts/run_alt_detectors.py --seed 20261017 --timestamp 2026-10-17T00:00:00+00:00\n```")
    return "\n".join(L) + "\n"
