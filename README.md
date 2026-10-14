# UAV Sentinel

**Telemetry Intelligence** — local-first UAV flight analysis with optional GPT-4o mini diagnostics.

| Phase | Capability |
|-------|------------|
| 1 | Upload & validate IMU / ATT / BARO |
| 2 | Conservative 12-feature LSTM scores + flags |
| 3 | Event grouping, Plotly dashboard, JSON/CSV reports |
| 4 | GPT-4o mini diagnostic report + flight-specific chat |

## Install & run

```bash
cd "/Users/purvansh/Desktop/ML Projects/UAV Telemetry"

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Detector artifacts live in ./uav_model/ (see ARTIFACTS.md)
# Edit .env and set OPENAI_API_KEY=sk-... for AI diagnostics

pytest -q
streamlit run app.py
```


## Phase 4 — AI diagnostics

- **LLM role:** interpretation only. Does not change scores, flags, threshold, or event grouping.
- **Model:** `OPENAI_MODEL` (default `gpt-4o-mini`).
- **Evidence:** compact pack from the Phase 3 report — detector config (exact 12 features), events, recon-error evidence, nearby baseline windows, data-quality warnings, and explicit BARO exclusion.
- **Outputs:** Pydantic-validated diagnostic JSON (`observed_facts` / hypotheses / unknowns / checks) plus optional attachment as `ai_analysis` on the flight report.
- **Chat:** bound to the active `flight_id` in Streamlit session state; switching flights resets chat.
- **Failures:** missing key / API errors leave detector results and exports available.

Routine `pytest` runs **mock** the OpenAI client — no paid calls.

## Event grouping (unchanged)

Merge flagged windows in the same session when intervals overlap or gap ≤ `max_gap_s` (default 1.5 s). Duration = `max(t_end)−min(t_start)`. Never merge across sessions.

## JSON report schema `1.0`

Detector summary + events + `data_quality_warnings` + `ai_analysis` (`null` or diagnostic object).

## Limitations

- AI hypotheses are not certified root causes.
- Do not commit `.env` or API keys.
