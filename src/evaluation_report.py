"""Run the reproducible detector evaluation and render JSON + Markdown reports.

Three strictly separate evidence classes:

* ``synthetic``   – injected perturbations on source flights (detector *sensitivity* only).
* ``real_world``  – reviewed intervals of real flights. Reported only when verified
  labels exist; otherwise status ``no_reviewed_labels`` and descriptive statistics only.
* ``descriptive`` – score distributions and flagged fractions of unlabelled flights.
  A flagged fraction is not a false-alarm rate.

No threshold is selected or promoted here. The deployed threshold is the baseline;
candidates (validation quantiles, a candidate calibration artifact) are compared with it.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from src.artifact_loader import ROOT, load_artifacts
from src.calibration import load_flight_frames, score_flight_frames
from src.data_quality import build_data_quality
from src.eval_corpus import resolve_flight
from src.detector import UAVDetector
from src.provenance import artifact_fingerprints
from src.reviewed_eval import (
    descriptive, evaluate as evaluate_reviewed, per_feature_error_summary, validate_intervals,
)
from src.split_manifest import validate_manifest
from src.synthetic_eval import (
    DEFAULT_LEVELS, INJECTION_TYPES, case_seed, detection_at_threshold, plan_injection,
    run_case, sigma_by_feature, summarize,
)
from src.threshold_sensitivity import candidate_thresholds_from_validation, score_distribution

SCHEMA = "evaluation_report/1"
SWEEP_FACTORS = (0.5, 0.75, 1.0, 1.25, 1.5)
REPS_PER_LEVEL = 2


def _git_state() -> Dict[str, Any]:
    def g(*a: str) -> str:
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return ""
    return {"commit": g("rev-parse", "HEAD") or None, "dirty": bool(g("status", "--porcelain", "--untracked-files=no"))}


def _resolve(flight: Mapping[str, Any]) -> Dict[str, pd.DataFrame]:
    return resolve_flight(flight)


def run_evaluation(
    manifest: Mapping[str, Any],
    *,
    seed: int,
    reviewed: Optional[Mapping[str, Any]] = None,
    candidate: Optional[Mapping[str, Any]] = None,
    out_dir: Optional[Path] = None,
    save_frames: bool = False,
    model_dir: Optional[Path] = None,
    levels: Mapping[str, List[Dict[str, Any]]] = DEFAULT_LEVELS,
    reps: int = REPS_PER_LEVEL,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    art = load_artifacts(model_dir)
    det = UAVDetector(art)
    sigma = sigma_by_feature(art)
    thr = art.threshold
    mv = validate_manifest(manifest)
    if not mv["ok"]:
        raise ValueError(f"split manifest invalid: {mv['errors']}")

    usable = [
        f for f in manifest["flights"]
        if f.get("split") in ("val", "test") and f.get("data_available") and f.get("duplicate_of") is None
    ]
    exclusions: List[Dict[str, Any]] = [
        {"flight_id": f["flight_id"], "reason": f.get("notes") or "excluded", "duplicate_of": f.get("duplicate_of")}
        for f in manifest["flights"] if f.get("split") == "excluded"
    ]

    base: Dict[str, Dict[str, Any]] = {}
    frames_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
    flights_out: List[Dict[str, Any]] = []
    for f in usable:
        fr = _resolve(f)
        frames_cache[f["flight_id"]] = fr
        res = score_flight_frames(fr, art, det, f["flight_id"])
        dq = build_data_quality(
            frames=fr, detail=res["detail"], deployed_features=list(art.features),
            candidate_features=list(res["detail"].cfg["features"]),
            used_source_columns={"IMU": ["GyrX", "GyrY", "GyrZ", "AccX", "AccY", "AccZ"],
                                 "ATT": ["Roll", "Pitch", "Yaw", "DesRoll", "DesPitch"],
                                 "BARO": ["Alt", "Press", "Temp"]},
            win_samples=art.win_samples, stride_samples=art.stride_samples,
            n_scored=len(res["windows"]),
        )
        base[f["flight_id"]] = res
        w = res["windows"]
        flights_out.append({
            "flight_id": f["flight_id"], "split": f["split"], "kind": f["kind"],
            "eval_role": f["eval_role"], "content_hash": f["content_hash"],
            "coverage_status": dq["coverage_status"],
            "flight_duration_s": dq["durations"]["flight_duration_s"],
            "analysed_duration_s": dq["durations"]["analysed_duration_s"],
            "windows_candidate": dq["windows"]["candidate"],
            "windows_scored": dq["windows"]["scored"],
            "windows_discarded": dq["windows"]["discarded"],
            "descriptive_at_deployed": descriptive(w, thr) if len(w) else None,
            "per_feature_error": per_feature_error_summary(w, thr) if len(w) else None,
        })
        if dq["windows"]["discarded"] or dq["coverage_status"] != "ok":
            exclusions.append({
                "flight_id": f["flight_id"], "reason": "partial coverage",
                "coverage_status": dq["coverage_status"],
                "windows_discarded": dq["windows"]["discarded"],
                "excluded_intervals": dq["excluded_interval_count"],
            })

    # ---------------- synthetic ----------------
    records: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    sources = [f for f in usable if f["eval_role"] == "synthetic_source"]
    syn_dir = (out_dir / "synthetic") if out_dir else None
    for f in sources:
        fid = f["flight_id"]
        b = base[fid]
        bw = b["windows"]
        for itype in INJECTION_TYPES:
            for li, lvl in enumerate(levels[itype]):
                for rep in range(reps):
                    iid = f"{fid}:{itype}:L{li}:r{rep}"
                    cs = case_seed(seed, fid, f["content_hash"], itype, li, rep)
                    rng = np.random.default_rng(cs)
                    if len(bw) == 0:
                        skipped.append({"injection_id": iid, "reason": "source has no scored windows"})
                        continue
                    spec = plan_injection(bw, itype, lvl, rng, cs, iid, thr)
                    if spec is None:
                        skipped.append({"injection_id": iid, "reason": "no clean placement (every candidate overlaps a flagged baseline window or leaves the session)"})
                        continue
                    rec, new_frames = run_case(frames_cache[fid], spec, flight_id=fid, base=b,
                                               artifacts=art, detector=det, sigma=sigma)
                    rec["split"] = f["split"]
                    rec["source_kind"] = f["kind"]
                    records.append(rec)
                    if save_frames and syn_dir is not None:
                        d = syn_dir / "frames" / iid.replace(":", "_")
                        d.mkdir(parents=True, exist_ok=True)
                        for s, df in new_frames.items():
                            df.to_parquet(d / f"{s}.parquet", index=False)
    if syn_dir is not None:
        syn_dir.mkdir(parents=True, exist_ok=True)
        slim = [{k: v for k, v in r.items() if k != "overlap_windows"} for r in records]
        (syn_dir / "injections.json").write_text(json.dumps(slim, indent=2, sort_keys=True, allow_nan=False) + "\n")

    def _syn(rs):
        return {"deployed": summarize(rs, thr)} if rs else {"status": "no injections"}

    synthetic = {
        "scope": "SYNTHETIC. Measures response to injected patterns; not real-world detection accuracy.",
        "seed": seed, "levels": levels, "reps_per_level": reps,
        "n_injections": len(records), "n_skipped": len(skipped), "skipped": skipped,
        "overall": _syn(records),
        "by_split": {s: _syn([r for r in records if r["split"] == s]) for s in ("val", "test")},
        "by_source_kind": {k: _syn([r for r in records if r["source_kind"] == k])
                           for k in sorted({r["source_kind"] for r in records})},
        "preprocessing_effects": {
            "n_pattern_preserved": sum(1 for r in records if r["pattern_preserved"]),
            "n_not_preserved": sum(1 for r in records if not r["pattern_preserved"]),
            "n_not_eligible": sum(1 for r in records if not r["eligible_for_scoring"]),
            "not_preserved_ids": [r["injection_id"] for r in records if not r["pattern_preserved"]],
            "not_eligible_ids": [r["injection_id"] for r in records if not r["eligible_for_scoring"]],
        },
        "collateral_flag_changes_total": sum(r["collateral_flag_changes_outside_interval"] for r in records),
        "unmatched_windows_outside_interval_total": sum(r["windows_outside_interval_without_baseline_counterpart"] for r in records),
        "flagged_unmatched_windows_outside_interval_total": sum(r["flagged_windows_outside_interval_without_baseline_counterpart"] for r in records),
        "flagged_unmatched_not_explained_by_baseline_flags_total": sum(r["flagged_unmatched_not_explained_by_baseline_flags"] for r in records),
        "interpretation": (
            "missing_values and dropout remove windows rather than perturb them: overlapping windows "
            "contain non-finite samples and are discarded (surfaced by the coverage summary), so the "
            "detector is never asked about them. They are 'not eligible', not 'missed'."
        ),
    }

    # ---------------- real-world (reviewed) ----------------
    reviewed = validate_intervals(reviewed) if reviewed else {"schema": "reviewed_intervals/1", "intervals": []}
    real_windows = {f["flight_id"]: base[f["flight_id"]]["windows"] for f in usable if f["kind"] == "real_unreviewed"}
    real = evaluate_reviewed(real_windows, reviewed, threshold=thr)
    real["scope"] = "REAL-WORLD, human-reviewed intervals only."
    real["labels_provenance"] = {"reviewer": reviewed.get("reviewer"), "reviewed_at": reviewed.get("reviewed_at"),
                                 "n_intervals": len(reviewed.get("intervals", []))}

    # ---------------- threshold comparison ----------------
    cand_thr: Dict[str, float] = {}
    val_elig = [f for f in usable if f["split"] == "val" and f.get("calibration_eligible") is True]
    val_scores = np.concatenate([base[f["flight_id"]]["scores32"] for f in val_elig]) if val_elig else np.array([])
    quantile_candidates: Dict[str, Any]
    if len(val_scores) >= 200:
        cand_thr.update({f"val_{k}": v for k, v in candidate_thresholds_from_validation(val_scores).items()})
        quantile_candidates = {"status": "computed", "n_validation_windows": int(len(val_scores)),
                               "flights": [f["flight_id"] for f in val_elig]}
    else:
        quantile_candidates = {"status": "not_computed",
                               "reason": "no calibration_eligible validation flights with >= 200 windows; no normal validation set exists in this checkout"}
    if candidate and candidate.get("threshold") is not None:
        cand_thr["candidate_artifact"] = float(candidate["threshold"])
    taus: Dict[str, float] = {"deployed": thr}
    taus.update({f"deployed_x{k:g}": thr * k for k in SWEEP_FACTORS if k != 1.0})
    taus.update(cand_thr)
    desc_windows = pd.concat([base[f["flight_id"]]["windows"] for f in usable], ignore_index=True) if usable else pd.DataFrame(columns=["anomaly_score"])
    comparison = {}
    for name, tau in taus.items():
        entry: Dict[str, Any] = {
            "threshold": float(tau),
            "synthetic_detection": detection_at_threshold(records, tau) if records else None,
            "descriptive_flagged_fraction_all_flights": (
                float((desc_windows["anomaly_score"] > tau).mean()) if len(desc_windows) else None),
            "descriptive_flagged_fraction_by_split": {
                s: (float((pd.concat([base[f["flight_id"]]["windows"] for f in usable if f["split"] == s])["anomaly_score"] > tau).mean())
                    if any(f["split"] == s for f in usable) else None)
                for s in ("val", "test")},
        }
        if real.get("status") == "ok":
            r = evaluate_reviewed(real_windows, reviewed, threshold=tau)
            entry["reviewed"] = {k: r[k] for k in ("event_recall", "false_alarms_per_flight_hour", "false_alarm_events",
                                                    "reviewed_unusual_intervals", "expected_hours_reviewed")}
        comparison[name] = entry
    threshold_comparison = {
        "baseline": "deployed",
        "selection_performed": False,
        "note": ("Descriptive sweep only. No threshold was chosen or promoted. Synthetic detection "
                 "rises as the threshold falls; without reviewed expected intervals the matching "
                 "false-alarm cost cannot be measured (flagged fractions are unreviewed)."),
        "validation_quantile_candidates": quantile_candidates,
        "candidate_artifact_status": (candidate or {}).get("status", "none"),
        "thresholds": comparison,
    }

    n_flights = len(usable)
    n_windows = int(sum(f["windows_scored"] for f in flights_out))
    report = {
        "schema": SCHEMA,
        "generated_at": timestamp or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": _git_state(),
        "model": {"artifacts_sha256": artifact_fingerprints(art), "features": list(art.features),
                  "window": [art.win_samples, art.stride_samples], "deployed_threshold": thr,
                  "threshold_source": "uav_model/threshold_config.json (recorded: 0.995 validation quantile; not reproduced — see artifact_audit.json)",
                  "flag_rule": "score > threshold"},
        "split_manifest": {"kind": manifest.get("kind"), "seed": manifest.get("seed"),
                           "validation": mv, "flights": [
                               {k: f.get(k) for k in ("flight_id", "split", "kind", "eval_role", "content_hash", "duplicate_of", "calibration_eligible")}
                               for f in manifest["flights"]]},
        "counts": {"flights_evaluated": n_flights, "windows_scored": n_windows,
                   "synthetic_injections": len(records),
                   "reviewed_intervals": len(reviewed.get("intervals", []))},
        "flights": flights_out,
        "data_quality_exclusions": exclusions,
        "synthetic": synthetic,
        "real_world": real,
        "descriptive": {"scope": "Unlabelled flights. Not accuracy, not false-alarm rates.",
                        "all_flights_at_deployed": descriptive(desc_windows, thr) if len(desc_windows) else None},
        "threshold_comparison": threshold_comparison,
        "limitations": limitations(manifest, real, quantile_candidates, synthetic, usable),
        "commands": COMMANDS,
    }
    return report


COMMANDS = [
    "python scripts/audit_artifacts.py",
    "python scripts/build_split_manifest.py --seed 20261017",
    "python scripts/calibrate_threshold.py --timestamp 2026-10-17T00:00:00+00:00",
    "python scripts/run_evaluation.py --seed 20261017 --timestamp 2026-10-17T00:00:00+00:00",
    "python scripts/verify_stored_flights.py",
    "python -m pytest -q",
]


def limitations(manifest, real, qc, synthetic, usable) -> List[str]:
    L = [
        "Historical calibration cannot be reproduced: the source flights, validation scores and "
        "calibration code are not in the repository (see outputs/evaluation/artifact_audit.json).",
        "The historical train/val/test split cannot be re-verified for duplicates or leakage; the "
        "evaluation split here is NEW and covers only the flights present in this checkout.",
    ]
    if qc.get("status") != "computed":
        L.append("No flight is declared representative/normal (calibration_eligible), so no validation "
                 "quantile threshold was computed and no candidate threshold exists to compare.")
    if real.get("status") != "ok":
        L.append("No human-reviewed real-world labels exist: event recall, false alarms per flight-hour, "
                 "specificity and precision are NOT measured and must not be quoted.")
    n_real = sum(1 for f in usable if f["kind"] == "real_unreviewed")
    L.append(f"Only {n_real} distinct real flight(s) are available as synthetic-injection sources; synthetic "
             "results describe sensitivity on those recordings and do not generalise to other drones or conditions.")
    L.append("Synthetic sources are unreviewed; a source flight is not claimed to be normal.")
    if synthetic["preprocessing_effects"]["n_not_preserved"]:
        L.append(f"{synthetic['preprocessing_effects']['n_not_preserved']} injection(s) were not preserved "
                 "by 10 Hz resampling (e.g. short spikes) and are reported separately.")
    L.append("Real flights flag a small share of windows without labels; those flags are unadjudicated.")
    return L


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def render_markdown(r: Mapping[str, Any]) -> str:
    L: List[str] = []
    m = r["model"]
    L.append("# Detector evaluation report\n")
    L.append(f"_Generated {r['generated_at']} · git `{(r['git']['commit'] or 'unknown')[:10]}`"
             f"{' (dirty)' if r['git']['dirty'] else ''}_\n")
    L.append("Scope: telemetry **anomaly detection** only. Synthetic and real-world results are reported "
             "separately and must not be mixed.\n")
    L.append("## Model and threshold\n")
    L.append(f"- Deployed threshold (baseline): `{m['deployed_threshold']!r}`; flag rule `{m['flag_rule']}`")
    L.append(f"- Threshold source: {m['threshold_source']}")
    for k, v in m["artifacts_sha256"].items():
        L.append(f"- sha256 {k}: `{v}`")
    L.append("\n## Data and splits\n")
    c = r["counts"]
    L.append(f"- Flights evaluated: **{c['flights_evaluated']}**, windows scored: **{c['windows_scored']}**, "
             f"synthetic injections: **{c['synthetic_injections']}**, reviewed intervals: **{c['reviewed_intervals']}**")
    L.append(f"- Split manifest: `{r['split_manifest']['kind']}` (seed {r['split_manifest']['seed']}), "
             f"validation ok: {r['split_manifest']['validation']['ok']}, counts {r['split_manifest']['validation']['counts']}\n")
    L.append("| flight | split | kind | coverage | windows scored / candidate | flagged @ deployed |")
    L.append("|---|---|---|---|--:|--:|")
    for f in r["flights"]:
        d = f["descriptive_at_deployed"] or {}
        L.append(f"| {f['flight_id']} | {f['split']} | {f['kind']} | {f['coverage_status']} | "
                 f"{f['windows_scored']} / {f['windows_candidate']} | {d.get('flagged', 0)} ({_pct(d.get('flagged_fraction'))}) |")
    L.append("\n### Exclusions and data-quality losses\n")
    for e in r["data_quality_exclusions"]:
        L.append(f"- `{e['flight_id']}`: {e['reason']}" + (f" (duplicate of `{e['duplicate_of']}`)" if e.get("duplicate_of") else "")
                 + (f" — coverage {e['coverage_status']}, {e['windows_discarded']} windows discarded" if "coverage_status" in e else ""))
    s = r["synthetic"]
    L.append("\n## Synthetic evaluation (NOT real-world accuracy)\n")
    L.append(f"{s['scope']} Seed {s['seed']}; {s['n_injections']} injections run, {s['n_skipped']} skipped (no clean placement).\n")
    ov = s["overall"].get("deployed", {}).get("overall") if "deployed" in s["overall"] else None
    if ov:
        L.append(f"At the deployed threshold: **{ov['n_detected']} / {ov['n_evaluable']}** evaluable injections detected "
                 f"({_pct(ov['detection_rate_eligible'])}); {ov['n_not_eligible']} not eligible for scoring; "
                 f"{ov['n_confounded_by_baseline']} confounded; median latency "
                 f"{'n/a' if ov['latency_s_median'] is None else format(ov['latency_s_median'], '.1f') + ' s'}.\n")
        L.append("| type | injections | evaluable | detected | rate | not eligible | median latency s |")
        L.append("|---|--:|--:|--:|--:|--:|--:|")
        for t, d in s["overall"]["deployed"]["by_type"].items():
            L.append(f"| {t} | {d['n_injections']} | {d['n_evaluable']} | {d['n_detected']} | {_pct(d['detection_rate_eligible'])} | "
                     f"{d['n_not_eligible']} | {'n/a' if d['latency_s_median'] is None else format(d['latency_s_median'], '.1f')} |")
        pp = s["overall"]["deployed"]
        L.append(f"\nPattern preserved by resampling: {pp['pattern_preserved'].get('n', 0)}; "
                 f"not preserved: {pp['pattern_not_preserved'].get('n', 0)}. Detection when preserved: "
                 f"{_pct(pp['pattern_preserved'].get('detection_rate_eligible'))}; when not preserved: "
                 f"{_pct(pp['pattern_not_preserved'].get('detection_rate_eligible'))}.")
    L.append(f"\nCollateral: {s['collateral_flag_changes_total']} flag change(s) outside injected intervals among windows common to the baseline run; "
             f"{s['unmatched_windows_outside_interval_total']} window(s) exist only in injected runs (window phase shift after a source-gap session split), "
             f"{s['flagged_unmatched_windows_outside_interval_total']} of them flagged "
             f"({s['flagged_unmatched_not_explained_by_baseline_flags_total']} not overlapping any baseline-flagged window).")
    L.append(f"\n{s['preprocessing_effects'].get('note', '')}{s['interpretation']}")
    rw = r["real_world"]
    L.append("\n## Real-world reviewed evaluation\n")
    if rw["status"] != "ok":
        L.append(f"**Not available.** {rw['reason']}\n")
    else:
        L.append(f"- Flights {rw['n_flights']}, reviewed unusual intervals {rw['reviewed_unusual_intervals']}, "
                 f"event recall {_pct(rw['event_recall'])}, false alarms/flight-hour "
                 f"{rw['false_alarms_per_flight_hour']}, missed {len(rw['missed_intervals'])}.")
    L.append("\n## Threshold comparison (no selection performed)\n")
    tc = r["threshold_comparison"]
    L.append(tc["note"] + "\n")
    L.append(f"Validation-quantile candidates: {tc['validation_quantile_candidates']['status']}"
             + (f" — {tc['validation_quantile_candidates'].get('reason', '')}" if tc['validation_quantile_candidates']['status'] != 'computed' else "") + "\n")
    L.append("| threshold | value | synthetic detected / evaluable | unreviewed flagged fraction (all flights) |")
    L.append("|---|--:|--:|--:|")
    for name, e in tc["thresholds"].items():
        sd = e["synthetic_detection"] or {}
        L.append(f"| {name}{' (baseline)' if name == 'deployed' else ''} | {e['threshold']:.4f} | "
                 f"{sd.get('n_detected', '–')} / {sd.get('n_evaluable', '–')} | {_pct(e['descriptive_flagged_fraction_all_flights'])} |")
    L.append("\n## Known limitations and missing evidence\n")
    for x in r["limitations"]:
        L.append(f"- {x}")
    L.append("\n## Reproduce\n")
    L.append("```bash")
    L.extend(r["commands"])
    L.append("```")
    return "\n".join(L) + "\n"
