"""UAV Sentinel FastAPI application."""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from backend.routers import flights, health  # noqa: E402

class QuietLocalModelProbes(logging.Filter):
    """Hide OpenCode's vLLM discovery polls against this API.

    OpenCode treats anything answering on 127.0.0.1:8000 as vLLM and
    repeatedly requests ``/health`` and ``/v1/models``. Those are not
    UAV Sentinel routes. ``/api/health`` is left visible.
    """

    _QUIET_PATHS = (
        '"GET /health ',
        '"HEAD /health ',
        '"GET /v1/models ',
        '"HEAD /v1/models ',
    )

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(path in message for path in self._QUIET_PATHS)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    logging.getLogger("uvicorn.access").addFilter(QuietLocalModelProbes())
    yield


app = FastAPI(
    title="UAV Sentinel API",
    description="Telemetry anomaly detection and AI diagnostics",
    version="1.0.0",
    lifespan=_lifespan,
)

cors_raw = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000",
)
origins = [o.strip() for o in cors_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(flights.router)


@app.exception_handler(Exception)
async def unhandled_exception(_request: Request, exc: Exception) -> JSONResponse:
    """Return JSON inside CORS so the browser can show the error instead of 'Failed to fetch'."""
    logging.getLogger("uvicorn.error").exception("Unhandled API error")
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})


@app.get("/")
def root():
    return {
        "service": "uav-sentinel",
        "docs": "/docs",
        "health": "/api/health",
    }


@app.get("/json/version")
def plotly_version_stub():
    """Plotly.js may probe this path; avoid noisy 404s when charts load."""
    return {"version": "0", "upgrade": False, "status": "ok"}
