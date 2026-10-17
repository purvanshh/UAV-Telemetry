"""Versioned Pydantic models for the UAV Sentinel flight analysis report.

Schema version ``1.0``. ``ai_analysis`` is null until Phase 4 attaches a
validated diagnostic object; detector fields remain immutable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SCHEMA_VERSION = "1.0"


class DetectorInfo(BaseModel):
    model: str = "LSTMAE"
    variant: str
    feature_count: int
    window_length: int
    stride: int
    threshold: float
    features: List[str] = Field(default_factory=list)
    feature_schema_version: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class FlightSummary(BaseModel):
    total_windows: int
    flagged_windows: int
    flagged_rate: Optional[float] = None
    event_count: int
    score_mean: Optional[float] = None
    score_p50: Optional[float] = None
    score_p95: Optional[float] = None
    score_max: Optional[float] = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _rate_consistency(self) -> "FlightSummary":
        if self.total_windows <= 0:
            if self.flagged_rate is not None:
                raise ValueError(
                    "flagged_rate must be null when total_windows is 0"
                )
        else:
            if self.flagged_rate is None:
                raise ValueError(
                    "flagged_rate required when total_windows > 0"
                )
            expected = self.flagged_windows / self.total_windows
            if abs(self.flagged_rate - expected) > 1e-9:
                raise ValueError(
                    f"flagged_rate {self.flagged_rate} != "
                    f"{self.flagged_windows}/{self.total_windows}"
                )
        return self


class EvidenceItem(BaseModel):
    type: str
    signal: Optional[str] = None
    peak_recon_error: Optional[float] = None
    mean_recon_error: Optional[float] = None
    n_samples: Optional[int] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None

    model_config = ConfigDict(extra="allow")


class AnomalyEventReport(BaseModel):
    event_id: str
    session_id: int
    start_time_s: float
    end_time_s: float
    duration_s: float
    peak_score: Optional[float] = None
    mean_flagged_score: Optional[float] = None
    flagged_window_count: int
    threshold: float
    peak_over_threshold: Optional[float] = None
    threshold_relative_score: Optional[float] = None
    window_indices: List[int] = Field(default_factory=list)
    affected_signals: List[str] = Field(default_factory=list)
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    data_quality_warnings: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    @field_validator("duration_s")
    @classmethod
    def _duration_non_neg(cls, v: float) -> float:
        if v < 0:
            raise ValueError("duration_s must be >= 0")
        return v

    @model_validator(mode="after")
    def _interval_order(self) -> "AnomalyEventReport":
        if self.end_time_s < self.start_time_s:
            raise ValueError("end_time_s must be >= start_time_s")
        if abs(self.duration_s - (self.end_time_s - self.start_time_s)) > 1e-6:
            raise ValueError("duration_s must equal end_time_s - start_time_s")
        if self.flagged_window_count != len(self.window_indices):
            raise ValueError(
                "flagged_window_count must equal len(window_indices)"
            )
        return self


class FlightReport(BaseModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    flight_id: str
    analysis_status: Literal[
        "completed", "insufficient_data", "error", "no_anomaly"
    ]
    detector: DetectorInfo
    summary: FlightSummary
    events: List[AnomalyEventReport] = Field(default_factory=list)
    data_quality_warnings: List[str] = Field(default_factory=list)
    grouping_policy: Dict[str, Any] = Field(default_factory=dict)
    ai_analysis: Optional[Any] = None
    errors: List[str] = Field(default_factory=list)
    # Optional additions (null in reports written before they existed):
    data_quality: Optional[Dict[str, Any]] = None  # src/data_quality.py
    provenance: Optional[Dict[str, Any]] = None  # src/provenance.py

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _event_count_matches(self) -> "FlightReport":
        if self.summary.event_count != len(self.events):
            raise ValueError(
                "summary.event_count must equal len(events)"
            )
        return self

    def to_json_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")


def validate_report(data: Union[Dict[str, Any], FlightReport]) -> FlightReport:
    if isinstance(data, FlightReport):
        return data
    return FlightReport.model_validate(data)
