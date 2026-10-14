# UAV Sentinel

**Telemetry Intelligence** — local-first UAV flight analysis with optional GPT-4o mini diagnostics.

| Layer | Stack |
|-------|--------|
| Frontend | Next.js (App Router) |
| Backend | FastAPI wrapping `src/` ML pipeline |
| Model | Conservative 12-feature LSTM AE in `uav_model/` |

## Install & run

### Backend

```bash
cd "/Users/purvansh/Desktop/ML Projects/UAV Telemetry"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Edit .env → OPENAI_API_KEY=sk-... (optional, for AI diagnostics)
uvicorn backend.main:app --reload --port 8000
```

API docs: http://127.0.0.1:8000/docs

### Frontend

```bash
cd frontend
npm install
# NEXT_PUBLIC_API_URL defaults to http://127.0.0.1:8000 via .env.local
npm run dev
```

UI: http://localhost:3000

### Tests

```bash
source .venv/bin/activate
pytest -q
```

## Workflow

1. **Upload** IMU / ATT / BARO CSVs or a ZIP → validated flight on disk (`data/flights/`)
2. **Analyze** → windows, scores, events, Plotly charts, JSON/CSV exports
3. **AI Diagnostics** → evidence pack → GPT-4o mini (scores/flags unchanged)
4. **Chat** → flight-bound follow-ups

## Environment

See `.env.example`:

- `OPENAI_API_KEY` / `OPENAI_MODEL` — Phase 4 AI (server-side only)
- `UAV_MODEL_DIR` — default `./uav_model`
- `DATA_DIR` — default `./data/flights`
- `CORS_ORIGINS` — default `http://localhost:3000,http://127.0.0.1:3000`

## Limitations

- AI hypotheses are not certified root causes.
- Do not commit `.env` or API keys.
- BARO altitude/pressure are excluded from LSTM scoring.
