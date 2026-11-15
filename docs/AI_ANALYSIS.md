# AI Flight Analysis — telemetry-aware interpretation

This document describes the pipeline behind **AI Analysis** (`/direct/[analysis_id]`). It is
separate from the deployed LSTM detector ("Deep Analysis"): the detector stays untouched and no
model result is used here.

## Why the old report was misleading

The previous pipeline was deliberately unit-free and relationship-free. It treated every
column as an unverified raw channel, detected departures independently per column, and then
grouped whatever happened to be nearby. The consequences, confirmed on `FLT_0003.CSV`:

1. **No physical meaning.** `t_ms` was only a hypothesis, so time was shown as raw ticks and
   seconds were conditional. `phi_deg`/`theta_deg` were "attitude-labelled channels", `gx` was
   "gyroscope-labelled", and accelerometer values were compared with no gravity context.
2. **No command/response comparison.** A motor spin-up caused by throttle, or an altitude
   change caused by the altitude setpoint, looked exactly like an unexplained deviation.
3. **Row numbers instead of flight time.** Reports led with `rows 397–459` and raw tick ranges.
4. **One blob per category.** All channels whose changes were near each other were merged into
   a single "statistical irregularity", mixing motor, IMU, altitude and controller terms, so the
   report could not say what kind of behaviour had changed.
5. **Status/aux/controller channels counted as aircraft behaviour.** An unknown `aux1` step or a
   small `u4` change could become a "period to review".
6. **Ratios against near-zero baselines.** "7.3× their earlier level" was computed from a
   0.01 m/s² baseline, which is numerically true and physically meaningless.
7. **The LLM was allowed to write, but not to interpret.** It received raw events and was
   forbidden from using units or physical direction, so it could only restate numbers.

## New architecture

```
CSV
 │
 ├─ 1. Schema & validation        config/telemetry_schema.json + src/telemetry_schema.py
 │      meaning · category · role · unit + evidence · validity · aliases
 │      units applied only when unit_status == "verified"
 │
 ├─ 2. Timing & data quality      src/direct_analysis.py
 │      ms→s, measured sample intervals, gaps, duplicates, missing, out-of-range, short files
 │
 ├─ 3. Signal inventory           schema-driven categories and roles
 │
 ├─ 4. Relationship analysis      src/flight_relationships.py
 │      roll/pitch command vs attitude · throttle vs motors · altitude estimate vs
 │      setpoint vs time-of-flight · attitude derivative vs gyro · |a| vs gravity ·
 │      motor balance · status transitions (reported separately, meanings not assumed)
 │
 ├─ 5. Event detection            src/event_engine.py
 │      robust rolling baseline, spike/sustained/variability/flatline/gap detectors,
 │      merged per physical family (attitude, acceleration, angular rate, altitude,
 │      motor, controller), minimum evidence, no ratio against zero
 │
 ├─ 6. Event interpretation       src/flight_relationships.py
 │      expected_response · unexplained_deviation · telemetry_inconsistency ·
 │      control_input_change · status_change · data_quality · insufficient_evidence
 │      plus physical before/during/peak values, command checks, sensor checks,
 │      significance floor and confidence
 │
 ├─ 7. Deterministic briefing     src/operator_report.py + src/pilot_report.py
 │      flight overview and sequence · what to check · what changed · what to do ·
 │      period cards · what was compared · overall behaviour · limitations
 │
 └─ 8. Optional AI narrative      src/direct_llm.py
        receives the structured analysis only; every number is grounded against it,
        units are allowed only for verified signals, and unsafe/unsupported sentences
        are removed and recorded
```

### Interpretation classes

| Class | Meaning |
|---|---|
| `expected_response` | The recorded input moved and the response follows the relationship measured across the whole recording. |
| `unexplained_deviation` | No recorded input or companion measurement explains the change. This is what a pilot should review. |
| `telemetry_inconsistency` | Two records of the same motion disagree (e.g. rotation rate without attitude change, or altitude sources apart). |
| `control_input_change` | A pilot-command or controller-term change; not an aircraft deviation by itself. |
| `status_change` | A status/flag transition. Encodings are not assumed; reported separately. |
| `data_quality` | Gap, missing values, flat-line or irregular sampling. |
| `insufficient_evidence` | Too small, too short, or no baseline to interpret. |

Command direction is **learned from the recording** (sign of the command/response correlation),
never assumed from an axis convention. The same applies to response labels; no "left/right" or
"nose-up/down" claim is made because axis signs are not established.

### Units policy

* Verified: `t_ms` → s, roll/pitch → deg, gyro → deg/s, accelerometer → m/s².
* Not verified (shown raw, never converted): motor outputs, RC commands, altitude estimate,
  setpoint, time-of-flight height/distance, controller terms, status flags.
* The LLM may use a unit only when the sentence references a channel whose schema unit exists;
  a unit attached to an unverified channel rejects that sentence (`unit_claim`).

## FLT_0003 example (9.2 s, 461 rows, 50 Hz)

What the new pipeline reports:

* `t_ms` verified as milliseconds; duration 9.2 s; sample rate 50.0 Hz; no gaps/missing rows.
* Sequence: arming status changes at 0:05; throttle input ramps from -1000 to -758 at 0:06;
  motor output leaves idle (1000) at 0:08 and reaches ~1580; altitude setpoint moves to 0.5 at 0:09.
* Relationships: throttle ↔ motor outputs consistent (r ≈ 0.70); altitude ↔ setpoint consistent;
  roll/pitch attitude change ↔ gyro consistent; |a| at rest 9.81 m/s² (gravity);
  roll/pitch *input* barely varies (range 2 raw counts) → tracking reported as insufficient data,
  not as a failure.
* One period stands out at 00:08–00:09: acceleration (peak 16.1 m/s²), rotation rate
  (±22 deg/s) and attitude (roll −2.2°, pitch +1.2°) change together and are **not explained
  by the recorded input**. Motor rise and altitude rise in the same window are marked
  **consistent with throttle/setpoint** and listed separately.
* Status transitions (arming, safety/aux/block flags) are reported as a count, not as
  flight-behaviour findings.

## Remaining limitations

* Axis/sign conventions are still not established, so no direction of motion is claimed.
* Motor, RC, altitude and controller units are undocumented; they stay raw.
* No yaw angle is present, so yaw motion cannot be interpreted.
* The system detects and explains unusual behaviour; it does not diagnose faults or judge safety.
* The interpretation is only as good as the recorded columns: a manoeuvre that is not represented
  in the RC channels (e.g. autonomous control) will look unexplained.
* Flight-phase detection (takeoff/cruise/landing) is not implemented; the sequence is derived
  from arming, throttle, motor level and setpoint changes.
