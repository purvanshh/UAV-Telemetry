"""Offline threshold-sensitivity report.

Does not modify uav_model/threshold_config.json or model weights.

Usage (from the repo root, venv active):

    python scripts/run_threshold_eval.py

Optional validation/test score table (not used to change the deployed threshold):

    UAV_EVAL_SCORES=data/eval/window_scores.parquet python scripts/run_threshold_eval.py

Required columns: flight_id, split, window_index, session_id, t_start, t_end, anomaly_score.
split must be 'val' or 'test'.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.artifact_loader import load_artifacts
from src.detector import UAVDetector
from src.fault_injection import injection_grid, score_aligned
from src.preprocessing_adapter import align_uploaded_flight
from src.threshold_sensitivity import (
    compare_unseen_to_reference,
    evaluate_score_table,
    load_score_table,
    missing_score_corpus,
    score_distribution,
    summarize_threshold,
)

OUT = ROOT / "outputs" / "threshold_eval"
FLIGHT_DIR = ROOT / "data" / "flights" / "flight-20261014T134921Z-38e246"


def _hist(scores: pd.Series, threshold: float, title: str, path: Path) -> None:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=scores, nbinsx=40, name="scores", marker_color="#5b8def"))
    fig.add_vline(x=threshold, line_dash="dash", line_color="#c9a227", annotation_text="deployed")
    fig.update_layout(
        title=title,
        template="plotly_dark",
        xaxis_title="reconstruction MSE",
        yaxis_title="windows",
        height=420,
    )
    fig.write_html(path, include_plotlyjs="cdn")


def _detection_plot(rows: pd.DataFrame, path: Path) -> None:
    if rows.empty:
        return
    fig = go.Figure()
    for feature, group in rows.groupby("feature"):
        # detection rate by magnitude, averaged over timing and duration
        rate = (
            group.groupby("magnitude_std")["detected_at_threshold"]
            .mean()
            .reset_index()
        )
        fig.add_trace(
            go.Scatter(
                x=rate["magnitude_std"],
                y=rate["detected_at_threshold"],
                mode="lines+markers",
                name=str(feature),
            )
        )
    fig.update_layout(
        title="Synthetic detection rate at the deployed threshold (not real-world accuracy)",
        template="plotly_dark",
        xaxis_title="injected offset (multiples of training std)",
        yaxis_title="fraction of injections detected",
        yaxis_range=[-0.05, 1.05],
        height=420,
    )
    fig.write_html(path, include_plotlyjs="cdn")


def _find_flight_frames() -> dict | None:
    if not (FLIGHT_DIR / "IMU.parquet").is_file():
        return None
    return {s: pd.read_parquet(FLIGHT_DIR / f"{s}.parquet") for s in ("IMU", "ATT", "BARO")}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    arts = load_artifacts()
    deployed = float(arts.threshold)
    missing = missing_score_corpus(arts.feature_schema)

    corpus_eval = None
    scores_path = os.environ.get("UAV_EVAL_SCORES", "")
    default_path = ROOT / "data" / "eval" / "window_scores.parquet"
    if not scores_path and default_path.is_file():
        scores_path = str(default_path)
    if scores_path and Path(scores_path).is_file():
        table = load_score_table(scores_path)
        corpus_eval = evaluate_score_table(table, deployed_threshold=deployed)
        missing["available"] = True
        missing["loaded_from"] = scores_path

    unseen = None
    synthetic_rows: list = []
    frames = _find_flight_frames()
    if frames is not None:
        aligned, _ = align_uploaded_flight(frames)
        detector = UAVDetector(arts)
        baseline = score_aligned(aligned, arts, detector)
        baseline["flight_id"] = "00000122"
        dist = score_distribution(baseline["anomaly_score"])
        at_deployed = summarize_threshold(baseline, deployed, features=list(arts.features))
        _hist(
            baseline["anomaly_score"],
            deployed,
            "Flight 00000122 window scores (unseen; not a fault label)",
            OUT / "flight_00000122_hist.html",
        )
        # Quiet anchor: start of a low-score window. Stressed: start of the max window.
        quiet = baseline.nsmallest(1, "anomaly_score").iloc[0]
        hot = baseline.nlargest(1, "anomaly_score").iloc[0]
        anchors = {
            "quiet": float(quiet["t_start"]),
            "near_peak": float(hot["t_start"]),
        }
        synthetic_rows = injection_grid(
            aligned=aligned,
            artifacts=arts,
            detector=detector,
            baseline_windows=baseline,
            features=("gyro_x", "acc_y", "roll"),
            magnitudes_in_std=(3.0, 8.0),
            durations_s=(3.0, 6.0),
            anchors=anchors,
            threshold=deployed,
        )
        syn = pd.DataFrame(synthetic_rows)
        syn.to_csv(OUT / "synthetic_faults.csv", index=False)
        _detection_plot(syn, OUT / "synthetic_detection.html")
        baseline.to_parquet(OUT / "flight_00000122_windows.parquet", index=False)
        unseen = {
            "flight_id": "00000122",
            "distribution": dist,
            "at_deployed_threshold": at_deployed,
            "anchors": anchors,
            "comparison_to_validation": None,
            "note": (
                "00000122 is not in feature_schema split lists. "
                "A score near the threshold is not a fault label."
            ),
        }
        if corpus_eval is not None:
            val_scores = load_score_table(scores_path)
            val_only = val_scores[val_scores["split"] == "val"]["anomaly_score"]
            unseen["comparison_to_validation"] = compare_unseen_to_reference(
                baseline["anomaly_score"], val_only
            )

    payload = {
        "deployed_threshold": deployed,
        "threshold_config": arts.threshold_config,
        "missing_corpus": missing,
        "corpus_eval": corpus_eval,
        "unseen_00000122": unseen,
        "synthetic_faults": synthetic_rows,
        "recommendation": (
            "Keep the deployed threshold 4.034964084625244. "
            "Validation window scores are not in this workspace, so a new quantile "
            "cannot be fit without using test data or the single unseen flight. "
            "Synthetic injections are not real-world accuracy and are not grounds "
            "to change production."
        ),
    }
    (OUT / "eval_summary.json").write_text(json.dumps(payload, indent=2, default=str))
    (OUT / "REPORT.md").write_text(_render(payload))
    print(f"wrote {OUT / 'REPORT.md'}")


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _render(payload: dict) -> str:
    missing = payload["missing_corpus"]
    lines = [
        "# Threshold sensitivity evaluation",
        "",
        "Offline report. The deployed threshold and model weights were not modified.",
        "",
        "## Deployed threshold",
        "",
        f"- Value: `{payload['deployed_threshold']}`",
        f"- Config: `{json.dumps(payload['threshold_config'])}`",
        "- Calibration code is not in this repository. The JSON records a 0.995 quantile on validation flights.",
        "",
        "## What is missing",
        "",
    ]
    if not missing.get("available"):
        lines.append("Saved validation and test window scores were not found.")
        lines.append("")
        lines.append(f"- Expected file: `{missing['required_file']}` (or `UAV_EVAL_SCORES`)")
        lines.append(f"- Columns: `{', '.join(missing['required_columns'])}`")
        lines.append(f"- Validation flight ids listed in the schema ({missing['n_val_flights_listed']}): `{', '.join(missing['val_flight_ids'])}`")
        lines.append(f"- Test flight ids listed in the schema ({missing['n_test_flights_listed']}): `{', '.join(missing['test_flight_ids'])}`")
        lines.append("")
        lines.append("Regenerate from the original flight CSVs with this checkpoint:")
        for item in missing["must_regenerate"]:
            lines.append(f"- {item}")
        lines.append("")
        lines.append("No labeled real anomalies are in this workspace, so event-level recall and false alarms per flight-hour cannot be calculated.")
    else:
        lines.append(f"Loaded score table from `{missing.get('loaded_from')}`.")
        ev = payload["corpus_eval"]
        lines.append("")
        lines.append("### Validation distribution")
        lines.append("")
        lines.append("```")
        lines.append(json.dumps(ev["validation_distribution"], indent=2))
        lines.append("```")
        lines.append("")
        lines.append("### Candidates fit on validation only")
        lines.append("")
        for name, summary in ev["validation_at_candidates"].items():
            lines.append(
                f"- **{name}** threshold={summary['threshold']:.6g}: "
                f"flagged {summary['n_flagged']}/{summary['n_windows']} "
                f"({summary['flagged_fraction']}), "
                f"flights with a flag {summary['n_flights_with_flag']}/{summary['n_flights']}, "
                f"events {summary['n_events']}"
            )
        if ev.get("test_at_deployed"):
            test = ev["test_at_deployed"]["deployed_only"]
            lines.append("")
            lines.append("### Held-out test at the deployed threshold only")
            lines.append("")
            lines.append(ev["test_at_deployed"]["note"])
            lines.append(
                f"- flagged {test['n_flagged']}/{test['n_windows']} "
                f"({test['flagged_fraction']}), "
                f"flights with a flag {test['n_flights_with_flag']}/{test['n_flights']}, "
                f"events {test['n_events']}"
            )
    lines.extend(["", "## Flight 00000122 (unseen example)", ""])
    unseen = payload.get("unseen_00000122")
    if not unseen:
        lines.append("Stored flight parquet was not found, so this example was skipped.")
    else:
        dist = unseen["distribution"]
        stats = unseen["at_deployed_threshold"]
        lines.append(unseen["note"])
        lines.append("")
        lines.append(
            f"- windows {dist['n']}, min {_fmt(dist['min'])}, p50 {_fmt(dist['p50'])}, "
            f"p95 {_fmt(dist['p95'])}, p99 {_fmt(dist['p99'])}, max {_fmt(dist['max'])}"
        )
        lines.append(
            f"- at deployed threshold: flagged {stats['n_flagged']}/{stats['n_windows']}, "
            f"events {stats['n_events']}"
        )
        lines.append(
            "- Not compared with a validation distribution because those scores are absent. "
            "Do not treat proximity of the max score to the threshold as a fault."
        )
        if unseen.get("comparison_to_validation"):
            lines.append("")
            lines.append("```")
            lines.append(json.dumps(unseen["comparison_to_validation"], indent=2))
            lines.append("```")
    lines.extend(
        [
            "",
            "## Synthetic fault injection",
            "",
            "Offsets were added to one aligned model feature for a fixed interval. "
            "These are synthetic results, not real-world diagnostic accuracy. "
            "No original fault-injection experiment exists in this repository; this grid is the reproducible substitute.",
            "",
        ]
    )
    rows = payload.get("synthetic_faults") or []
    if not rows:
        lines.append("Synthetic grid was not run.")
    else:
        detected = sum(1 for r in rows if r.get("detected_at_threshold"))
        increased = sum(1 for r in rows if r.get("intended_feature_error_increased"))
        lines.append(f"- injections: {len(rows)}")
        lines.append(f"- detected at the deployed threshold: {detected}/{len(rows)}")
        lines.append(f"- intended feature reconstruction error increased: {increased}/{len(rows)}")
        lines.append("")
        lines.append("| feature | timing | stds | duration_s | detected | overlap max score | intended error up | outside flags |")
        lines.append("|---|---|---:|---:|---|---:|---|---:|")
        for r in rows:
            lines.append(
                f"| {r['feature']} | {r['timing']} | {r['magnitude_std']} | {r['duration_s']} | "
                f"{r['detected_at_threshold']} | {_fmt(r.get('overlap_score_max_injected'))} | "
                f"{r['intended_feature_error_increased']} | {r['outside_windows_flagged']} |"
            )
    lines.extend(
        [
            "",
            "## Trade-off",
            "",
            "A lower quantile flags more validation windows and more flights, which is a higher false-alarm rate if those flights are treated as normal. A higher quantile misses more borderline windows. Flight 00000122's maximum (about 4.006) sits just under the stored 0.995 threshold. Moving the threshold to catch that window would be a policy change, and it cannot be justified from one unlabeled flight.",
            "",
            "Real-anomaly recall cannot be established: this workspace has no labeled anomaly events.",
            "",
            "## Recommendation",
            "",
            payload["recommendation"],
            "",
            "## Rerun",
            "",
            "```bash",
            "source .venv/bin/activate",
            "python scripts/run_threshold_eval.py",
            "```",
            "",
            "Plots: `outputs/threshold_eval/flight_00000122_hist.html`, `outputs/threshold_eval/synthetic_detection.html`.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
