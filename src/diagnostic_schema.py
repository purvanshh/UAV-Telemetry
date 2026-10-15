"""Pydantic contract for GPT-4o mini diagnostic responses."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


Confidence = Literal["low", "medium", "high"]


class CauseHypothesis(BaseModel):
    hypothesis: str
    confidence: Confidence
    supporting_evidence: List[str] = Field(default_factory=list)
    contradicting_evidence: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class EventFinding(BaseModel):
    event_id: str
    possible_causes: List[CauseHypothesis] = Field(default_factory=list)
    observed_facts: List[str] = Field(default_factory=list)
    unknowns: List[str] = Field(default_factory=list)
    recommended_checks: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class DiagnosticReport(BaseModel):
    """Structured AI interpretation — never mutates detector scores/flags."""

    flight_id: str
    summary: str
    findings: List[EventFinding] = Field(default_factory=list)
    overall_limitations: List[str] = Field(default_factory=list)
    insufficient_evidence: bool = False
    model_name: Optional[str] = None
    notes: Optional[str] = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("summary")
    @classmethod
    def _summary_nonempty(cls, v: str) -> str:
        if not str(v).strip():
            raise ValueError("summary must be non-empty")
        return v.strip()

    def to_json_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")


class ChatAnswer(BaseModel):
    """Structured follow-up chat reply."""

    flight_id: str
    answer: str
    referenced_event_ids: List[str] = Field(default_factory=list)
    caveats: List[str] = Field(default_factory=list)
    insufficient_evidence: bool = False

    model_config = ConfigDict(extra="ignore")

    @field_validator("answer")
    @classmethod
    def _answer_nonempty(cls, v: str) -> str:
        if not str(v).strip():
            raise ValueError("answer must be non-empty")
        return v.strip()

    @field_validator("caveats", "referenced_event_ids", mode="before")
    @classmethod
    def _coerce_str_list(cls, v: Any) -> List[str]:
        """Models often return a single string instead of a list."""
        if v is None:
            return []
        if isinstance(v, str):
            text = v.strip()
            return [text] if text else []
        if isinstance(v, (int, float)):
            return [str(v)]
        return v

    def to_json_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")


def validate_diagnostic(data: Union[Dict[str, Any], DiagnosticReport]) -> DiagnosticReport:
    if isinstance(data, DiagnosticReport):
        return data
    return DiagnosticReport.model_validate(data)


def validate_chat_answer(data: Union[Dict[str, Any], ChatAnswer]) -> ChatAnswer:
    if isinstance(data, ChatAnswer):
        return data
    return ChatAnswer.model_validate(data)


def insufficient_evidence_report(
    flight_id: str,
    *,
    reason: str,
    model_name: Optional[str] = None,
) -> DiagnosticReport:
    return DiagnosticReport(
        flight_id=flight_id,
        summary=(
            "Insufficient evidence to propose a specific physical fault. "
            + reason.strip()
        ),
        findings=[],
        overall_limitations=[
            "The LSTM flags reconstruction anomalies on 12 IMU/ATT features only.",
            "BARO altitude/pressure/temperature are excluded from the deployed detector.",
            "An anomaly score is not a confirmed root-cause diagnosis.",
            reason.strip(),
        ],
        insufficient_evidence=True,
        model_name=model_name,
        notes="Generated locally without asserting a fault hypothesis.",
    )
