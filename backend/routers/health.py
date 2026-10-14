"""Health and model artifact status."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from backend.schemas import ApiConfiguredResponse, ArtifactsStatus, HealthResponse
from src.artifact_loader import ArtifactError, load_artifacts, missing_artifacts, resolve_model_dir
from src.llm_service import api_configured, get_model_name

router = APIRouter(tags=["system"])


@router.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@router.get("/api/artifacts/status", response_model=ArtifactsStatus)
def artifacts_status() -> ArtifactsStatus:
    model_dir = resolve_model_dir()
    missing = missing_artifacts(model_dir)
    if missing:
        return ArtifactsStatus(
            ready=False,
            missing=missing,
            model_dir=str(model_dir),
        )
    try:
        arts = load_artifacts(model_dir)
        return ArtifactsStatus(
            ready=True,
            missing=[],
            model_dir=str(model_dir),
            variant=arts.variant,
            threshold=arts.threshold,
            features=list(arts.features),
            n_features=arts.n_features,
            win_samples=arts.win_samples,
            stride_samples=arts.stride_samples,
            warnings=list(arts.warnings),
        )
    except ArtifactError as exc:
        return ArtifactsStatus(
            ready=False,
            missing=missing,
            model_dir=str(Path(model_dir)),
            error=str(exc),
        )


@router.get("/api/ai/status", response_model=ApiConfiguredResponse)
def ai_status() -> ApiConfiguredResponse:
    return ApiConfiguredResponse(configured=api_configured(), model=get_model_name())
