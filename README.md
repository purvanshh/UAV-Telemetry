# UAV Sentinel

Local-first telemetry intelligence for UAV flights. Upload IMU / ATT / BARO streams, score a conservative LSTM autoencoder, inspect flagged events, export reports, and optionally ask GPT-4o mini for evidence-grounded diagnostics.

| Layer | Stack |
|-------|--------|
| UI | Next.js App Router (`frontend/`) |
| UI (local alternative) | Streamlit (`app.py`, `pages/`) |
| API | FastAPI (`backend/`) wrapping the ML pipeline in `src/` |
| Detector | 12-feature LSTM Autoencoder artifacts in `uav_model/` |
| Store | Disk-backed flights under `data/flights/` (gitignored) |
| Offline evaluation | `src/` + `scripts/` + `evaluation/` manifests (no server needed) |

No cloud account is required for detection. OpenAI is optional and only used for AI diagnostics and chat.

---

## Architecture

```text
IMU / ATT / BARO CSVs or ZIP
   │
   ▼
Upload validation            backend/routers/flights.py · src/upload.py · src/validation.py
   │   · schema / column checks                 src/schema.py
   │   · one timestamp interpretation           src/timebase.py
   │   · duplicate detection (content hash)     src/flight_identity.py
   ▼
Flight store                 backend/store.py → data/flights/{id}/ (meta.json + parquet frames)
   │
   ▼
Analysis pipeline            src/inference_service.py
   │   validation → 10 Hz stream alignment      src/preprocessing_adapter.py
   │   → 30/15 windows                          src/window_generator.py
   │   → LSTM AE scoring                        src/detector.py + uav_model/
   │   → flag iff score > threshold, group events   src/event_builder.py
   │   → FlightReport                           src/report_schema.py
   │   + data-quality coverage                  src/data_quality.py
   │   + feature lineage & artifact fingerprints    src/provenance.py
   │
   ├──► Next.js UI (frontend/)        analysis, coverage panel, charts, AI page, reports
   ├──► Streamlit UI (app.py, pages/)    same pipeline, local alternative
   └──► AI diagnostics (optional)     evidence pack → GPT-4o mini (src/llm_service.py)
        detector scores/flags are immutable; output is pilot-plain language

Offline (no server): evaluation & calibration layer
   src/split_manifest.py · src/eval_corpus.py · src/synthetic_eval.py ·
   src/reviewed_eval.py · src/calibration.py · src/alt_detectors.py
   → docs/EVALUATION.md · docs/ALT_DETECTORS.md
```

---

## What's new — last 5 commits

| Commit | Change |
|--------|--------|
| `8cfc0c8` | **Pipeline hardening.** One timestamp interpretation shared by validation and preprocessing (`src/timebase.py`); deterministic duplicate detection at upload — byte-identical / partial-overlap / similar (`src/flight_identity.py`); per-analysis data-quality coverage (`src/data_quality.py`); feature lineage, artifact fingerprints and run provenance in reports (`src/provenance.py`). |
| `51353bb` | **Evaluation harness + calibration.** Flight-level split manifests with leakage checks (`evaluation/manifests/`), evaluation corpus with dedup/exclusions, reviewed-interval and synthetic evaluation, and candidate-only threshold calibration (`scripts/calibrate_threshold.py`) — the deployed threshold is never auto-changed. See [docs/EVALUATION.md](docs/EVALUATION.md). |
| `abcc0c1` | **Coverage & duplicates surfaced; deployed model documented.** Upload responses report duplicate matches (`duplicate_of`, `allow_duplicate` override); analyze responses carry coverage status + detail, shown in both UIs; [docs/DEPLOYED_MODEL.md](docs/DEPLOYED_MODEL.md) is generated from `uav_model/` with a `--check` drift test; threshold citations corrected to `4.034964084625244`. |
| `e76d036` | **Context / consistency features.** `src/context_features.py` derives Euler-rate ↔ gyro consistency, commanded-vs-actual tracking and coarse flight phases (side detectors only — the LSTM input stays fixed); `alt_rate`/`press_rate` provenance units now state what preprocessing actually computes. |
| `bedca88` | **Alternative detectors (Phase 3/4).** Window-statistic, consistency, command-tracking and phase-aware side detectors evaluated against the deployed LSTM on a matched alert burden, with a pre-registered synthetic screen. Status and caveats: [docs/ALT_DETECTORS.md](docs/ALT_DETECTORS.md). |

---

## Quick start

### 1. Backend

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # optional: set OPENAI_API_KEY for AI features
uvicorn backend.main:app --reload --port 8000
```

- API: http://127.0.0.1:8000  
- Docs: http://127.0.0.1:8000/docs  
- Health: http://127.0.0.1:8000/api/health  

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

- UI: http://localhost:3000  
- API URL defaults to `http://127.0.0.1:8000` (`frontend/.env.local` → `NEXT_PUBLIC_API_URL`)

### 3. Streamlit UI (optional, local alternative)

```bash
pip install streamlit        # not included in requirements.txt
streamlit run app.py
```

### 4. Tests

```bash
source .venv/bin/activate
pytest -q
```

---

## UI map

| Section | Route | What it does |
|---------|-------|----------------|
| **Overview** | `/` | Saved flights, status counts, open / delete |
| **New Analysis** | `/upload` | Upload three CSVs or a ZIP; validate and open the flight |
| **Flight Analysis** | `/flights/[id]` | Run LSTM scoring, metrics, data-coverage panel, score timeline, event charts |
| **AI Diagnostics** | `/flights/[id]/ai` | Evidence pack → GPT-4o mini report + flight-bound chat |
| **Reports** | `/flights/[id]/reports` | JSON report, windows CSV, events CSV, trace preview |

Flights persist on disk. After the first upload, Overview lists them so you do not need to re-upload.

The Streamlit app mirrors this flow locally: `app.py` → `pages/1_Flight_Analysis.py`, `2_AI_Diagnostics.py`, `3_Reports.py`.

---

## Workflow

1. **Ingest** — Provide synchronized `IMU.csv`, `ATT.csv`, and `BARO.csv` (or a ZIP that contains them). The API validates schema and timestamps, flags exact/partial duplicates (override with `allow_duplicate=true`), and stores the flight under `data/flights/{id}/`.
2. **Analyze** — Sliding windows (30 samples, stride 15) are scored by the LSTM AE. Flag iff `score > threshold`. Events are grouped contiguous flags. Every run also records data-coverage accounting (rows kept, gaps, windows scored vs candidate) and provenance (feature lineage, artifact fingerprints) in the report.
3. **Inspect** — Plotly charts for score timeline, per-feature reconstruction error, and BARO context (context only; not scored).
4. **Diagnose (optional)** — Build an evidence pack from detector outputs and call GPT-4o mini. Scores, flags, and events are never changed by the model.
5. **Chat** — Follow-ups receive the diagnostic JSON plus detector evidence as context.
6. **Export** — Download flight report JSON, windows CSV, and events CSV.

---

## Detector (deployed)

| Setting | Value |
|---------|--------|
| Architecture | LSTM AE · hidden 48 · latent 12 · 1 layer |
| Features | 12 conservative IMU/ATT signals |
| Window / stride | 30 / 15 samples (~10 Hz alignment) |
| Score | Mean squared reconstruction error after scaler |
| Threshold | `4.034964084625244` (recorded as validation quantile 0.995; **not reproducible from this repo** — see [docs/DEPLOYED_MODEL.md](docs/DEPLOYED_MODEL.md)) |
| Flag rule | `score > threshold` (equality is not a flag) |

Authoritative, generated description of the deployed features, scaler, window and threshold: [docs/DEPLOYED_MODEL.md](docs/DEPLOYED_MODEL.md).

**Scored features:** `gyro_x/y/z`, `acc_x/y/z`, `roll`, `pitch`, `yaw_sin`, `yaw_cos`, `roll_res`, `pitch_res`

**Not scored:** BARO altitude / pressure / temperature. BARO may appear in charts and reports as context only.

Windows never cross session gaps (source timestamp gaps &gt; ~1 s).

Reproducibility: `python scripts/verify_stored_flights.py` re-scores every stored flight and compares per-window scores, flags and times against the saved windows (exit 1 on drift; never writes to the store).

Artifacts expected in `uav_model/`:

- `lstm_autoencoder.pth` (or a valid extracted Torch zip folder)
- `scaler.json`, `model_config.json`, `feature_schema.json`, `threshold_config.json`
- `preprocessing.py`, `inference.py`

---

## AI diagnostics

- Model: `gpt-4o-mini` by default (`OPENAI_MODEL`)
- Key: `OPENAI_API_KEY` in project `.env` (server-side only; never exposed to the browser)
- The LLM receives detector evidence and the prior diagnostic JSON for chat. It must not invent telemetry, alter scores/flags, or treat hypotheses as verified faults.
- Output is written for the pilot, not an engineer: everyday words ("unusualness score", "alert level", "unusual stretch"), no ML/statistics jargon, and every number anchored against the alert level. The detector vocabulary and guardrails are enforced by `src/diagnostic_prompt.py`.
- Without a key, upload / analyze / export still work; Generate diagnostic and chat stay disabled.

---

## Environment

Copy `.env.example` → `.env`:

| Variable | Purpose | Default |
|----------|---------|---------|
| `OPENAI_API_KEY` | AI diagnostics + chat | unset |
| `OPENAI_MODEL` | Chat model name | `gpt-4o-mini` |
| `UAV_MODEL_DIR` | Model artifacts | `./uav_model` |
| `DATA_DIR` | Flight store | `./data/flights` |
| `CORS_ORIGINS` | Allowed UI origins | `localhost:3000` / `127.0.0.1:3000` |
| `UAV_MAX_UPLOAD_MB` | Upload size hint | `100` |

Frontend:

| Variable | Purpose | Default |
|----------|---------|---------|
| `NEXT_PUBLIC_API_URL` | FastAPI base URL | `http://127.0.0.1:8000` |

Do not commit `.env` or API keys.

---

## Project layout

```
├── backend/            # FastAPI app, flight store, routers
├── frontend/           # Next.js UI
├── app.py + pages/     # Streamlit UI (local alternative)
├── src/                # Validation, timebase, preprocessing, detector, events,
│                       #   data quality, provenance, evaluation, LLM
├── uav_model/          # Trained artifacts + preprocessing/inference helpers
├── docs/               # DEPLOYED_MODEL.md (generated) · EVALUATION.md · ALT_DETECTORS.md
├── evaluation/         # Split manifests + reviewed-interval ground truth
├── scripts/            # Offline tools: eval runs, calibration, audits, verification
├── tests/              # Pytest suite (API, pipeline, evaluation, AI schema)
├── data/flights/       # Local flight store (gitignored contents)
└── outputs/            # Eval / scratch outputs (gitignored)
```

---

## API snapshot

| Method | Path | Notes |
|--------|------|-------|
| `GET` | `/api/health` | Liveness |
| `GET` | `/api/artifacts/status` | Model readiness |
| `GET` | `/api/ai/status` | Whether OpenAI is configured |
| `GET/POST` | `/api/flights` | List / upload (duplicate detection; `allow_duplicate` to override) |
| `POST` | `/api/flights/{id}/analyze` | Run detector (returns `coverage_status` + `data_quality`) |
| `GET` | `/api/flights/{id}/events` | Grouped events |
| `GET` | `/api/flights/{id}/charts/{name}` | Plotly JSON |
| `POST` | `/api/flights/{id}/ai/diagnostic` | Generate AI report |
| `POST` | `/api/flights/{id}/ai/chat` | Follow-up chat |
| `GET` | `/api/flights/{id}/exports/...` | JSON / CSV downloads |

Full OpenAPI: `/docs`.

---

## Limitations

- Anomaly scores are reconstruction MSE, not calibrated fault probabilities.
- AI output is interpretive only — not a flight-safety or go/no-go authority.
- No labeled real-world anomaly corpus ships with this repo. The threshold's recorded provenance (0.995 quantile on validation flights) cannot be re-derived here: the source flights, validation scores and calibration script are absent. See [docs/EVALUATION.md](docs/EVALUATION.md).
- Alternative detectors / context features are offline research tools — not part of the deployed scoring path — and their screen is synthetic-only until reviewed real labels exist. See [docs/ALT_DETECTORS.md](docs/ALT_DETECTORS.md).
- Local OpenAI-compatible tools that probe `127.0.0.1:8000` as a vLLM host are unrelated to this API.

---

## License / use

Intended for local research and engineering review of UAV telemetry. Keep secrets out of git and treat detector flags as statistical exceedances, not confirmed hardware faults.
