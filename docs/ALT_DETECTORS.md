# Alternative detectors and context features (Phases 3–4)

Baseline = the deployed aggregate LSTM score with its deployed threshold (unchanged).
`python scripts/run_alt_detectors.py` writes `outputs/evaluation/alt_detectors/report.{json,md}`.
Protocol, thresholds and the pass rule are in the docstring of `src/alt_eval.py` and are copied into the report.

**Status: the success criteria are NOT demonstrated on real data.** There are no reviewed real labels, so
additional *labelled real* anomalies cannot be counted and false alarms cannot be measured. What exists is
a synthetic screen with a matched alert-burden proxy. Treat its passes as hypotheses.

## Detectors

| group | detector | idea |
|---|---|---|
| baseline | `lstm_aggregate` | deployed score > 4.034964 |
| 3 | `lstm_per_feature` | max over features of LSTM error ÷ that feature's reference Q99 |
| 3 | `zmax_w30/w10`, `diffe_w30/w10`, `flat_w30/w10` | window statistics: max \|z\|, first-difference energy, flatline; 3 s and 1 s windows |
| 4 | `rate_consistency_w30` | RMS(Euler-derived body rate − gyro) |
| 4 | `cmd_tracking_w30` | RMS of commanded−actual roll/pitch/yaw |
| 4 | `fused_lstm_rate`, `fused_lstm_cmd`, `fused_all` | max of Q99-normalised components |
| 4 | `fused_lstm_rate_phase` | same, normalised per coarse flight phase |

Revised window length applies only to the side detectors; the LSTM input is fixed at 30 samples and was not retrained.

## Comparable alert burden

Reference statistics are fitted on the burden set (unreviewed real flights in `val`) and each alternative's
threshold is set so it flags the same fraction of those windows as the baseline does at the deployed threshold.
On held-out `test` sources, burden is re-measured as events per analysed hour and flagged-window fraction
(both must stay ≤ 1.25× baseline). These are unreviewed alerts — **not** false-alarm rates.

## Verified conventions (Phase 4)

- ATT angles are degrees (ZYX Euler), gyro is rad/s body rate. Euler→body-rate uses the full ZYX rotation.
  Verified label-free on both real flights: corr 0.71–0.97, slope 0.78–1.18; the rotation beats the naive
  derivative on pitch (0.91 vs 0.84, 0.71 vs 0.66). The gate (corr ≥ 0.7, slope 0.6–1.6 per axis per real flight)
  must pass or the rate detectors are excluded. `114` pitch is borderline (0.71).
- `alt_rate`/`press_rate` in the aligned frame are per-grid-step differences, not per second (schema says
  m/s). Not scored by the model; provenance now states this.
- Flight phase uses net altitude change per window and gyro RMS, fixed constants (0.5 m/s, 0.1 rad/s). A first
  draft used the mean of 10 Hz altitude differences, which labelled ~88 % of a real flight "vertical" (BARO noise);
  it was replaced on that evidence, before the comparison was re-run. Phase labels are not independently verified.

## Reading the result

See the generated report. Caveats that matter: one real burden flight (520 windows), two held-out sources
(one real, one generated), held-out phases `steady`/`active` only, 40 evaluable injections, and injections that
perturb gyro or attitude alone directly break the consistency check — favourable to it by construction. Phase
conditioning added gains but also flagged more windows and failed the burden rule.
