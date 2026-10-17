# Evaluation and threshold calibration

Scope: telemetry **anomaly detection** only — no fault diagnosis or root-cause claims.
What is deployed is described in [DEPLOYED_MODEL.md](DEPLOYED_MODEL.md). This page
describes how the detector and its threshold are evaluated, and what the repository
can and cannot prove.

## What is verified about the deployed threshold

`python scripts/audit_artifacts.py` writes `outputs/evaluation/artifact_audit.json`.

| Verified from files / git | Recorded in JSON only (not verified) | Cannot be recovered |
|---|---|---|
| `lstm_ae = 4.034964084625244` (exactly a float32 value); file unchanged since the commit that added it | "0.995 quantile" (`quantile_used`) | the 101 source flights |
| strict `score > threshold` in deployed code | "calibrated on validation flights only" | the validation scores |
| `scaler.json` fit on exactly the 71 flights in `split.train` | | the calibration script / quantile implementation |
| split id lists pairwise disjoint | | duplicate/near-duplicate checks of the historical split |

Conclusion: **the deployed threshold's provenance is incomplete.** It is used as the
baseline; it is not claimed to be a reproduced 99.5th-percentile value. (An earlier PRD
draft cited `4.42494`; that number matches no artifact and was corrected.)

## Splits and leakage

`evaluation/manifests/` holds machine-readable manifests (`split_manifest/1`):

- `historical_split.json` — ids copied from `feature_schema.json`; `kind: historical_unverified`.
- `evaluation_split.json` — a **new** split over the flights in this checkout (real flights in
  `data/flights/`, generated cases in `testdata/`). Not a reproduction of the historical split.

Rules (`src/split_manifest.py`, enforced by tests): a flight has one split, so windows never cross
splits; byte-identical or identical-content recordings may not sit in different splits (duplicates are
`excluded` with `duplicate_of`); shared-stream near-duplicates across splits are reported. Duplicate
detection uses `src/flight_identity.py` (file SHA-256 + canonical content hash).
Regenerate: `python scripts/build_split_manifest.py --seed 20261017`.

## Calibration (candidate only)

`python scripts/calibrate_threshold.py [--quantile 0.995] [--timestamp ISO]`

Loads the deployed checkpoint and scaler, scores every flight with `split == "val"` **and**
`calibration_eligible == true` using the deployed semantics, and takes
`numpy.quantile(float32 scores, q, method="linear")` over all windows pooled. Writes
`outputs/calibration/threshold_candidate.json` with threshold, quantile, method, window and flight
counts, flight ids, checkpoint/scaler/preprocessing hashes, preprocessing config, timestamp, score
distribution and a comparison with the deployed value. It never writes `uav_model/threshold_config.json`;
promotion is a manual decision. Test flights are never opened.

`calibration_eligible` is a human declaration that a flight is representative behaviour; no tool sets it.
Absent such flights the script reports `no_validation_data` and exits 2.

## Evaluation

`python scripts/run_evaluation.py [--reviewed ...] [--candidate ...] [--seed 20261017] [--timestamp ISO]`
writes `outputs/evaluation/evaluation_report.{json,md}` and `synthetic/injections.json`.
Three evidence classes are never mixed:

1. **Synthetic** (`src/synthetic_eval.py`): seeded perturbations (spike, drift, stuck, missing values, dropout,
   oscillation, multi-feature) injected into *raw* frames of source flights, never modifying the source.
   Each injection records source flight, columns, parameters, exact interval, affected rows, seed, whether
   10 Hz resampling preserved the pattern, and whether a scored window overlapped it (eligible). Detection is
   per injected event: any overlapping scored window above the threshold; placement requires clean baseline
   windows so a flag is attributable. Missing values/dropouts remove windows (reported through coverage),
   so they are *not eligible*, not "missed". **Not real-world accuracy.**
2. **Real-world reviewed** (`src/reviewed_eval.py`, labels in `evaluation/manifests/reviewed_intervals.json`):
   labels `expected | unusual_telemetry | data_quality_issue | uncertain`, with reviewer + date. Windows are
   grouped into events (overlapping windows are one event). Recall = unusual intervals overlapped by ≥ 1 event;
   false alarm = event overlapping `expected` time and nothing else; false alarms per flight-hour use the
   reviewed expected hours as denominator; unreviewed time is never normal. The shipped file is empty — no
   verified labels exist, so these metrics are reported as unavailable.
3. **Descriptive**: score distributions and flagged fractions of unlabelled flights (not false-alarm rates).

Threshold comparison keeps the deployed value as baseline, adds validation-quantile candidates when an
eligible validation set exists, and a descriptive sweep; nothing is selected or promoted.

Alternative detectors and context features: [ALT_DETECTORS.md](ALT_DETECTORS.md).

## Reproduce everything

```bash
python scripts/audit_artifacts.py
python scripts/build_split_manifest.py --seed 20261017
python scripts/calibrate_threshold.py --timestamp 2026-10-17T00:00:00+00:00   # exit 2 until eligible val flights exist
python scripts/run_evaluation.py --seed 20261017 --timestamp 2026-10-17T00:00:00+00:00
python scripts/verify_stored_flights.py      # stored flights re-score to their saved windows
python scripts/run_alt_detectors.py --seed 20261017 --timestamp 2026-10-17T00:00:00+00:00
python scripts/describe_deployment.py --check
python -m pytest -q
```

`data/flights/` is git-ignored; the evaluation manifest records content hashes so a different checkout is
detectable (`calibrate_threshold.py` refuses data whose hash differs from the manifest).
