# FLT_*.CSV (`uav_flt_csv` v1.2.0) — evidence, open questions, and what can be analysed

Reproduce every number here:

```
python scripts/analyze_flt_physics.py tests/fixtures/data/FLT_0001.CSV tests/fixtures/data/FLT_0002.CSV
python scripts/check_training_convention.py "<path to the ArduPilot-extracted 'uav dataset' folder>"
python scripts/confirm_profile_facts.py --list
```

`tests/test_flt_evidence.py` pins them. **Status: blocked from the deployed detector** (no yaw, no desired roll/pitch).

## 0. Corrections to profile v1.1.0 (found in review)

| v1.1.0 claim | Problem | Now |
|---|---|---|
| "Negate all three accelerometer axes" (`accel_polarity = −1`) | Only the (roll_sign, pitch_sign) = (+1, +1) member of a four-way family. With a ROS-style frame (roll +1, pitch −1) it would corrupt `ax`, `gy` and pitch. It was derived from the sign of `az` alone. | Signs are **derived** from owner choices `roll_sign`, `pitch_sign` (§3). Nothing is proposed. |
| Rest levels "predict ax, ay, az to 0.02" as polarity evidence | Rest roll/pitch of such loggers are normally computed *from* the accelerometer, so the agreement is partly by construction (§2). The pitch tilt is only 1.17°. | Reported as unit/formula evidence; polarity needs a tilt test. |
| Numbers (gyro slopes, rest levels) | Quoted from ad-hoc runs the script did not reproduce (different smoothing; script used whole-file medians). | Script now produces all of them; a test checks the quoted rest numbers against it. |
| `gyro_unit` confirmation also verified `gz` | `gz` has no evidence. | `gz` has its own facts (`gyro_z_unit`, `gyro_z_sign`). |
| Confirmations bound to profile version only; CLI silently skipped changes | Same version number could hide an edited claim; no audit trail. | Per-fact digest, mandatory author + basis, append-only log with supersede/revoke (§5). |
| Duplicate upload of a barometer-less flight was stored twice | Identity hash needed all three streams. | Fixed (§6). Data-quality showed the IMU-timestamp scaffold as 599 barometer rows. Fixed (§6). |

## 1. Provenance

No data dictionary exists. The files arrived by WhatsApp (quarantine attribute, no source URL). `~/Downloads/LOGS*` are the
ArduPilot logs of the *training* data (a different aircraft and logger); the notebooks there are unrelated.

## 2. Unit evidence vs. frame/polarity evidence

### Units (strong where noted)

| Fact | Evidence | Strength |
|---|---|---|
| accel is **m/s²** | median \|a\| 9.82 / 9.80 (ratio to g 1.002 / 0.999), all three axes jointly | strong |
| angles are **degrees** | at rest FLT_0001: φ=−4.56°, θ=−1.17°; the accel shows (+0.198, −0.780, +9.791) vs the degree projection (+0.200, −0.780, +9.774). Radians would predict ay≈+7 m/s². | strong *for the unit*; see circularity |
| **time ms / gyro deg/s** (a pair) | d(φ)/dt vs gx, t as ms: gain bounded 0.85–1.14 (FLT_0001), 0.83–1.61 (FLT_0002). Rules out s/µs (×1000) and rad/s (×57). | strong for the *family*; scale only to ≈±40% |
| gz unit | none | **none** |

**Circularity.** At rest the logged roll/pitch are almost certainly computed from the accelerometer
(φ = atan2(ay, az), θ = atan2(−ax, √(ay²+az²)) reproduces them), so agreement shows the angle unit and the firmware's formula, not an independent
check of axis direction or accelerometer bias. Each file has exactly **one** rest segment (250 samples, motors idle); only FLT_0001 is tilted
(φ −4.56°, θ −1.17°); FLT_0002 is level. In flight a multirotor's accelerometer measures thrust, not tilt, so the gravity test is
valid only at rest. The **gyro–angle** relation is the independent one (separate sensors, dynamic).

### Gyro `gy` vs pitch rate (weaker)

Always positive (r 0.64–0.90), but the gain band is wide and, in FLT_0001, **excludes 1.0** (0.51–0.92). Cumulative check: change in pitch vs ∫gy has slope
0.32 / 0.25 (r 0.77 / 0.44). The *training* logs show the same shape — roll gain 53.6–57.5, pitch 49.8–60.2 deg/s per rad/s — because attitude is an
estimator output (accelerometer-corrected), not the gyro integral. So the attenuation is not by itself a unit error, and it does not move the
deg/s-vs-rad/s decision (×57). What it does mean: **the exact scale of `gy` is not established**, so the unit is `supported`, never `verified`, until the owner says so.

### Frame / polarity

What the data does fix, given the training convention (measured: AccZ ≈ −9.85, AccY ≈ −g·sinRoll with r = 0.98, gyro +57 deg/s per rad/s):

* `az′ = −az` — direct, in both files.
* `ay′ = −roll_sign·ay`, `gx′ = roll_sign·gx`, `roll′ = roll_sign·φ`
* `ax′ = −pitch_sign·ax`, `gy′ = pitch_sign·gy`, `pitch′ = pitch_sign·θ`

What it cannot fix: **`roll_sign` and `pitch_sign`.** All four combinations reproduce every observation (`tests/test_frame_model.py` generates the
file under each and recovers the model frame exactly). They are physical facts (which wing/nose is "positive" depends on how the IMU is mounted).
`ax`'s own polarity is not independently measured (one 1.17° tilt, and θ is probably computed from `ax`). `gz` polarity cannot be tested (no yaw angle).

Why `az ≈ +9.8` vs `−9.7`: the logger's accelerometer reads gravity with positive sign when upright; the training
logs follow ArduPilot FRD. Negating all three axes (the v1.1.0 rule) is right only for (+1, +1); a ROS-style frame
(x forward, y left, z up, right-handed angles ⇒ roll +1, pitch −1) needs `ax ×(+1), ay ×(−1), az ×(−1), gy ×(−1), pitch ×(−1)`.

## 3. Facts and what settles them

| Fact | Kind | Data support | Settled by |
|---|---|---|---|
| `time_unit = ms` | unit | strong (family) | owner confirmation |
| `angle_unit = deg` | unit | strong | owner confirmation |
| `accel_unit = m/s²` | unit | strong | owner confirmation |
| `gyro_unit = deg/s` (gx, gy) | unit | strong for family, scale ±40% | owner confirmation |
| `az_polarity = −1` | frame | direct (rest) | owner confirmation |
| `roll_sign`, `pitch_sign` ∈ {+1, −1} | frame choice | **none** (4-way tie) | **tilt test** (raise nose, lower right wing) or documentation |
| `gyro_z_unit`, `gyro_z_sign` | unit/frame | **none** | logger documentation |

**Sufficiently supported for explicit owner confirmation:** `time_unit`, `angle_unit`, `accel_unit`, `gyro_unit` (gx, gy), `az_polarity`.
**Still need a physical test or logger documentation:** `roll_sign`, `pitch_sign`, `gyro_z_unit`, `gyro_z_sign`.

## 4. Missing model inputs (unchanged conclusion)

| Needed | In the file? | Derivable? |
|---|---|---|
| measured **yaw** | no (`rc_yaw` is a pilot command) | No: integrating `gz` is rejected (unit/polarity untestable, drift, changes the training preprocessing). |
| **desired roll/pitch** | no (`rc_roll/rc_pitch` are stick commands) | No. |
| barometer | no | Not a model input (§6). `tof_h = tof_mm/1000` (r = 1.000), `z_est ≈ tof_h` (r ≥ 0.995): rangefinder heights, not mapped to `alt`. |

**The deployed detector cannot analyse these files, and confirming every fact above would not change that.** No fabricated yaw, desired attitude,
barometer or convention is used anywhere. A dataset-specific detector (Path B) is not feasible with ~22 s / ≈11 windows of unlabelled data; none is built and no accuracy is claimed.

## 5. Confirmation log

`config/profile_confirmations.json` is append-only and **does not exist yet** (nothing has been confirmed). Each record: `action`, profile id **and** version,
`fact`, `value`, `confirmed_by`, `basis` (both mandatory), `recorded_at`, `fact_digest` (hash of the fact's definition and the evidence of every column it governs),
`evidence_snapshot`, `supersedes`. A record is ignored — and reported — if the profile version or digest changed, if the value is not allowed, if author/basis is missing, or if
it names another profile. Later records supersede; `--revoke` appends a revocation; nothing is overwritten. Confirmations are read only by the combined-file profile path:
three-file uploads, the model artifacts and stored flights are untouched (tested).

## 6. Optional barometer (review of the change)

Pinned on the real artifacts (`tests/test_optional_baro.py`): window scores/flags/sessions/events are bit-identical for any barometer values (real, NaN, absurd); the barometer's
*timestamps* do bound the grid (a late-starting barometer shortens the windows), so the opt-in scaffold copies **IMU** timestamps, which provably equals a barometer with full coverage;
IMU gaps still split sessions; a barometer-less flag is inert when real barometer data exists; a real barometer under an optional profile equals the three-file flow; partial barometer columns are ignored with a
warning; three-file uploads still require BARO. Defects found and fixed: duplicate detection (identity hash needed three streams), data-quality counted the scaffold as 599 barometer rows.
