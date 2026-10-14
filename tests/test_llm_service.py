"""LLM service — mocked API, no paid calls."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from src.llm_service import (
    LLMConfigError,
    LLMResponseError,
    LLMService,
    OpenAIChatClient,
    attach_ai_to_report,
    api_configured,
)
from src.report_builder import build_flight_report

FEATURES = [
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "acc_x",
    "acc_y",
    "acc_z",
    "roll",
    "pitch",
    "yaw_sin",
    "yaw_cos",
    "roll_res",
    "pitch_res",
]


class FakeResponse:
    def __init__(self, content: str):
        self.choices = [
            SimpleNamespace(message=SimpleNamespace(content=content))
        ]
        self.usage = SimpleNamespace(prompt_tokens=10, completion_tokens=20)


class FakeClient:
    def __init__(self, payload: dict):
        self.payload = payload
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse(json.dumps(self.payload))


def _report(with_events=True):
    windows = pd.DataFrame(
        [
            {
                "window_index": i,
                "session_id": 0,
                "t_start": float(i * 1.5),
                "t_end": float(i * 1.5 + 2.9),
                "anomaly_score": 5.0 if with_events and i == 1 else 1.0,
                "threshold": 4.0,
                "is_anomaly": bool(with_events and i == 1),
            }
            for i in range(4)
        ]
    )
    return (
        build_flight_report(
            flight_id="llm-1",
            analysis_status="completed",
            windows=windows,
            threshold=4.0,
            variant="conservative",
            features=FEATURES,
        ),
        windows,
    )


def test_generate_diagnostic_mocked():
    report, windows = _report(True)
    payload = {
        "flight_id": "llm-1",
        "summary": "One flagged interval with elevated gyro reconstruction error.",
        "findings": [
            {
                "event_id": report.events[0].event_id,
                "possible_causes": [
                    {
                        "hypothesis": "Unusual gyroscope behavior",
                        "confidence": "low",
                        "supporting_evidence": ["peak score above threshold"],
                        "contradicting_evidence": [],
                    }
                ],
                "observed_facts": ["Flagged window count >= 1"],
                "unknowns": ["No actuator telemetry"],
                "recommended_checks": ["Inspect gyro axes"],
            }
        ],
        "overall_limitations": ["Conservative 12-feature model"],
        "insufficient_evidence": False,
    }
    client = FakeClient(payload)
    service = LLMService(client=client, model="gpt-4o-mini")
    diag = service.generate_diagnostic(report, windows=windows)
    assert diag.flight_id == "llm-1"
    assert diag.findings
    assert client.calls
    assert client.calls[0]["response_format"] == {"type": "json_object"}
    assert service.last_usage is not None
    assert service.last_usage.prompt_tokens == 10

    combined = attach_ai_to_report(report, diag)
    assert combined.ai_analysis is not None
    assert combined.summary.flagged_windows == report.summary.flagged_windows


def test_no_events_still_calls_model_with_score_context():
    """Zero-event flights must reach the LLM with aggregate score evidence."""
    report, windows = _report(False)
    payload = {
        "flight_id": "llm-1",
        "summary": (
            "No windows exceeded the saved threshold "
            "(score_max well below threshold). No detector events to diagnose."
        ),
        "findings": [
            {
                "event_id": "flight_overview",
                "possible_causes": [],
                "observed_facts": [
                    "flagged_windows=0",
                    "event_count=0",
                    "score_max below threshold",
                ],
                "unknowns": ["Actuator commands not in evidence pack"],
                "recommended_checks": [
                    "Compare against a known anomalous flight if available"
                ],
            }
        ],
        "overall_limitations": [
            "Conservative 12-feature LSTM; BARO excluded",
            "Below-threshold windows are not confirmed anomalies",
        ],
        "insufficient_evidence": True,
    }
    client = FakeClient(payload)
    service = LLMService(client=client, model="gpt-4o-mini")
    diag = service.generate_diagnostic(report, windows=windows)
    assert client.calls, "API must be called for zero-event flights"
    user_msg = client.calls[0]["messages"][1]["content"]
    assert "score_context" in user_msg
    assert "top_windows_by_score" in user_msg
    assert "unflagged_high_score_context_only" in user_msg or "below_threshold" in user_msg
    assert diag.insufficient_evidence is True
    assert service.last_evidence_pack is not None
    assert service.last_evidence_pack["summary"]["flagged_windows"] == 0
    assert service.last_evidence_pack["events"] == []


def test_insufficient_data_still_local():
    report, _ = _report(False)
    report = report.model_copy(update={"analysis_status": "insufficient_data"})
    client = FakeClient({"should": "not be called"})
    service = LLMService(client=client)
    diag = service.generate_diagnostic(report)
    assert diag.insufficient_evidence is True
    assert client.calls == []


def test_malformed_json_raises():
    report, windows = _report(True)

    class BadClient:
        def create(self, **kwargs):
            return FakeResponse("not-json{{{")

    service = LLMService(client=BadClient())
    with pytest.raises(LLMResponseError):
        service.generate_diagnostic(report, windows=windows)


def test_schema_mismatch_raises():
    report, windows = _report(True)
    client = FakeClient({"flight_id": "llm-1", "summary": ""})  # empty summary
    service = LLMService(client=client)
    with pytest.raises(LLMResponseError):
        service.generate_diagnostic(report, windows=windows)


def test_missing_api_key_config_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert api_configured() is False
    with pytest.raises(LLMConfigError):
        OpenAIChatClient(api_key="")


def test_chat_mocked():
    report, windows = _report(True)
    payload = {
        "flight_id": "llm-1",
        "answer": "A sensor issue is possible but not established.",
        "referenced_event_ids": [report.events[0].event_id],
        "caveats": ["Hypothesis only"],
        "insufficient_evidence": False,
    }
    service = LLMService(client=FakeClient(payload))
    ans = service.chat(
        report=report, user_message="Could this be a sensor issue?", windows=windows
    )
    assert "sensor" in ans.answer.lower()
