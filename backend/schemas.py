"""API request/response models for UAV Sentinel FastAPI."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "uav-sentinel"


class ArtifactsStatus(BaseModel):
    ready: bool
    missing: List[str] = Field(default_factory=list)
    model_dir: str
    variant: Optional[str] = None
    threshold: Optional[float] = None
    features: List[str] = Field(default_factory=list)
    n_features: Optional[int] = None
    win_samples: Optional[int] = None
    stride_samples: Optional[int] = None
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = None


class FlightListItem(BaseModel):
    flight_id: str
    status: str
    source: Optional[str] = None
    created_at: Optional[str] = None
    filenames: Dict[str, str] = Field(default_factory=dict)


class UploadResponse(BaseModel):
    ok: bool
    flight_id: Optional[str] = None
    status: Optional[str] = None
    source: Optional[str] = None
    filenames: Dict[str, str] = Field(default_factory=dict)
    file_sizes: Dict[str, int] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    validation: Optional[Dict[str, Any]] = None


class AnalyzeRequest(BaseModel):
    max_gap_s: float = 1.5


class AnalyzeResponse(BaseModel):
    ok: bool
    flight_id: str
    status: str
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    n_windows: int = 0
    n_flagged: int = 0
    flagged_rate: Optional[float] = None
    n_events: int = 0
    threshold: float = 0.0
    variant: str = ""
    features: List[str] = Field(default_factory=list)
    score_mean: Optional[float] = None
    score_p50: Optional[float] = None
    score_p95: Optional[float] = None
    score_max: Optional[float] = None


class FlightDetail(BaseModel):
    flight_id: str
    status: str
    source: Optional[str] = None
    filenames: Dict[str, str] = Field(default_factory=dict)
    file_sizes: Dict[str, int] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    validation: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None
    analyzed_at: Optional[str] = None
    analysis: Optional[Dict[str, Any]] = None
    has_report: bool = False
    has_ai_diagnostic: bool = False
    event_count: int = 0


class DiagnosticRequest(BaseModel):
    selected_event_ids: Optional[List[str]] = None


class ChatRequest(BaseModel):
    message: str
    selected_event_id: Optional[str] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatStateResponse(BaseModel):
    flight_id: str
    messages: List[ChatMessage] = Field(default_factory=list)
    selected_event_id: Optional[str] = None


class ApiConfiguredResponse(BaseModel):
    configured: bool
    model: str
