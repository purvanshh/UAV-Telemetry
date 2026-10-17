# UAV Sentinel

Local-first telemetry intelligence for UAV flights. Upload IMU / ATT / BARO streams, score a conservative LSTM autoencoder, inspect flagged events, export reports, and optionally ask GPT-4o mini for evidence-grounded diagnostics.

| Layer | Stack |
|-------|--------|
| UI | Next.js App Router (`frontend/`) |
| API | FastAPI (`backend/`) wrapping the ML pipeline in `src/` |
| Detector | 12-feature LSTM Autoencoder artifacts in `uav_model/` |
| Store | Disk-backed flights under `data/flights/` (gitignored) |

No cloud account is required for detection. OpenAI is optional and only used for AI diagnostics and chat.

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

### 3. Tests

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
| **Flight Analysis** | `/flights/[id]` | Run LSTM scoring, metrics, score timeline, event charts |
| **AI Diagnostics** | `/flights/[id]/ai` | Evidence pack → GPT-4o mini report + flight-bound chat |
| **Reports** | `/flights/[id]/reports` | JSON report, windows CSV, events CSV, trace preview |

Flights persist on disk. After the first upload, Overview lists them so you do not need to re-upload.

---

## Workflow

1. **Ingest** — Provide synchronized `IMU.csv`, `ATT.csv`, and `BARO.csv` (or a ZIP that contains them). The API validates schema and stores the flight under `data/flights/{id}/`.
2. **Analyze** — Sliding windows (30 samples, stride 15) are scored by the LSTM AE. Flag iff `score > threshold`. Events are grouped contiguous flags.
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
├── backend/           # FastAPI app, flight store, routers
├── frontend/          # Next.js UI
├── src/               # Validation, windows, detector, reports, LLM
├── uav_model/         # Trained artifacts + preprocessing/inference helpers
├── tests/             # Pytest suite (API, detector, evidence, chat schema)
├── scripts/           # Offline eval helpers (e.g. threshold sensitivity)
├── data/flights/      # Local flight store (gitignored contents)
└── outputs/           # Eval / scratch outputs (gitignored)
```

---

## API snapshot

| Method | Path | Notes |
|--------|------|-------|
| `GET` | `/api/health` | Liveness |
| `GET` | `/api/artifacts/status` | Model readiness |
| `GET` | `/api/ai/status` | Whether OpenAI is configured |
| `GET/POST` | `/api/flights` | List / upload |
| `POST` | `/api/flights/{id}/analyze` | Run detector |
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
- Local OpenAI-compatible tools that probe `127.0.0.1:8000` as a vLLM host are unrelated to this API.

---

## License / use

Intended for local research and engineering review of UAV telemetry. Keep secrets out of git and treat detector flags as statistical exceedances, not confirmed hardware faults.
