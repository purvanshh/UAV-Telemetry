"""Health and AI status."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter

from src.llm_client import api_configured, get_model_name

router = APIRouter(tags=["system"])


@router.get("/api/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "service": "arjuna"}


@router.get("/api/ai/status")
def ai_status() -> Dict[str, Any]:
    return {"configured": api_configured(), "model": get_model_name()}
