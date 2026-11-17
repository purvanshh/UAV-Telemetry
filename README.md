# ARJUNA — flight log analysis

Upload an ARJUNA flight log (`flight.ulg`, or a zipped flight folder) and get:

1. **Code findings** for every fault code in the ARJUNA Flight Logging Standard v1.1, using the thresholds Labhesh
   approved, plus physics checks from the quadcopter model. These come back in about a second, with no AI.
2. **An agentic investigation.** An AI agent works through code tools: it verifies each finding on the raw signals,
   looks for problems the rules missed, orders events into a cause chain and answers the pilot's "what I saw"
   report. Every number it writes was computed by a tool and is checked before it is shown.

**Every CSV is converted to the standard first.** `src/arjuna/convert.py` turns it into a real ARJUNA v1.1 `.ulg`
(SI units, radians, FLU body frame, hash chain), which then goes through exactly the same pipeline as a native log.
The FLT format (33 columns, 50 Hz) has an exact mapping in `config/arjuna/csv_formats.json`. Any other CSV is mapped
by column name, with the time unit, angle and gyro units, g vs m/s², and an FRD→FLU frame flip all inferred and
written down. Columns that cannot be mapped are dropped and listed, never guessed. Where the CSV allows, the converter
also derives standard records: vibration (`accel_vibe`, the ArduPilot VIBE method) from the accelerometer, and an
altitude-loop `arjuna_controller` record from setpoint vs estimate. The conversion report (mapping, derived records,
assumptions, dropped columns) is shown on the page, and both the converted `.ulg` and the original CSV can be
downloaded.

FLT specifics, decided by the owner: these are short test hops, so there is no minimum air time: every recording that gets airborne is
analysed as a flight (a bench test, with no movement, is still flagged). Which corner each motor (m1–m4) sits on is unknown, so
motors are named m1–m4 and no position-based conclusion is drawn.

The earlier LSTM "Deep Analysis", the Streamlit app and the separate statistical CSV engine have been removed. They
are in git history.

## How it follows the ARJUNA guide

| Guide step | Where |
|---|---|
| 1. Import with pyulog (no hand-written parser) | `src/arjuna/ulog.py` |
| 2. Verify the hash chain (Verified / Verified up to t / Modified) | `verify_chain` in `src/arjuna/ulog.py` |
| 3. Refuse unknown `schema_version`, frames or profile | `check_compatibility` |
| 4. One timeline; dropouts and missing sensors marked, gaps never filled; missing = NaN, never 0 | `src/arjuna/ulog.py`, `src/arjuna/analysis.py` |
| 5. Rule checks for all 13 fault codes, with each code reported as checked, events-only or not checkable | `src/arjuna/checks.py` |
| Physics: hover balance, thrust residual, per-motor η least-squares fit | `src/arjuna/physics.py` |
| 6. Findings with code, severity, window, detector, confidence and evidence | `run_checks` / `analyse` |
| AI rules: grounded, cited, no LLM arithmetic, causes as suggestions, "I don't know" | `src/agent/prompt.py`, `src/agent/validate.py` |
| Pilot report: symptom prior × time proximity × confidence | `src/agent/symptoms.py` |
| Privacy: GPS coordinates and engineer labels are never sent to the AI | `src/agent/sources.py` |

Thresholds: `config/arjuna/thresholds.json` (approved 16 Nov 2026). Enum tables: `config/arjuna/enums_provisional.json`.
Airframe: `config/arjuna/airframe.json`. Checks with no approved threshold yet say so and are marked `confidence: medium`.

**Validation on the 263-flight ARJUNA dataset:** the checks reproduce every one of Labhesh's labels on the 188 real
flights (0 missed across 8 fault codes), and every flight is classified the same way as the triage
(flight / bench test / ground only). The η fit recovers a simulated 20%-weak motor to within 1%.

## How the agent works

- The overview, the findings and the physics are fetched by code first, so coverage never depends on the model.
- `submit_report` is withheld until the agent has examined every finding (its time window or its evidence signals)
  and made a minimum number of investigation calls.
- Tools: `signal_stats`, `compare_periods`, `find_periods`, `dominant_frequency`, `correlate`,
  `controller_tracking`, `motor_outputs`, `get_messages`, `sequence_of_departures` (what changed first, before an
  event), `state_at`, `get_physics` and `match_pilot_report`.
- The validator removes any sentence whose numbers do not trace to a tool result, any safety verdict and any cause
  stated as certain. Findings the agent skipped are added back as "not reviewed".
- Model: `UAV_AGENT_MODEL` (default `gpt-5.4-mini`). On the dataset, gpt-4o-mini skipped verification and
  misordered causes; gpt-5.4-mini followed the protocol. One investigation takes about 30–90 s and 50–200k input
  tokens.

## Run locally

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env            # OPENAI_API_KEY=..., optionally UAV_AGENT_MODEL=gpt-5.4-mini
.venv/bin/uvicorn backend.main:app --reload --port 8000

cd frontend && npm install
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 npm run dev    # http://localhost:3000
```

Tests: `.venv/bin/python -m pytest -q` (synthetic ARJUNA logs are written with Labhesh's ULog writer,
`src/arjuna/ulog_writer.py`, which the CSV converter also uses).

## API

| Method | Path | |
|---|---|---|
| POST | `/api/analyses` | upload `.ulg` / flight-folder `.zip` / `.csv` (converted first), optional `symptom`, `time_ref`, `time_value`, `note`; code only |
| POST | `/api/analyses/{id}/agent` | `{"consent": true}`: run the investigation |
| GET | `/api/analyses`, `/api/analyses/{id}` | list, view |
| GET | `/api/analyses/{id}/findings/{fid}/context` | chart data for a finding |
| GET | `/api/analyses/{id}/series?sig=topic:field:instance` | chart data for any signal |
| GET | `/api/analyses/{id}/download/ulg`, `/download/original` | the standard log the analysis read; the CSV as uploaded |
| GET | `/api/analyses/{id}/export` | JSON export |
| GET | `/api/health`, `/api/ai/status` | |

## Environment

| Variable | |
|---|---|
| `OPENAI_API_KEY` | required for the agent |
| `UAV_AGENT_MODEL` | agent model, default `gpt-5.4-mini` |
| `ANALYSIS_DATA_DIR` | where analyses are stored (default `data/analyses`) |
| `CORS_ORIGINS`, `CORS_ORIGIN_REGEX` | allowed frontend origins |
| `UAV_MAX_UPLOAD_MB` | upload limit (default 100 MB) |

## Deploy

Backend: Hugging Face Docker Space via `scripts/deploy_hf.sh <user>/<space>`. Frontend: Vercel with root
`frontend` and `NEXT_PUBLIC_API_URL` set to the Space URL.

## Known limits

- The dataset is converted ArduPilot data at 10–25 Hz. The η dynamics fit needs body rates and motor outputs at
  ≥ 100 Hz (ARJUNA engineering profile), so on these logs it reports why it did not run and the hover balance is
  used instead. TIMING and SECURITY need topics only the ARJUNA FC produces.
- Signatures: the hash chain is verified; the RSA signature is reported as present but not checked until a board
  public key is configured.
- The guide asks for an offline-first assistant. This deployment uses a cloud model, with consent per flight, and
  sends only tool results.
