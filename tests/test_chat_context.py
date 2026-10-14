"""Flight chat isolation."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from src.chat_service import ask_followup, empty_chat_state, ensure_chat_for_flight
from src.diagnostic_schema import DiagnosticReport
from src.llm_service import LLMService
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
        self.usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1)


class FakeClient:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        # Echo flight_id from a fixed good answer
        payload = {
            "flight_id": "wrong-id",  # model mislabels — service should correct via chat_service
            "answer": "Answer grounded in context.",
            "referenced_event_ids": [],
            "caveats": [],
            "insufficient_evidence": False,
        }
        return FakeResponse(json.dumps(payload))


def _mini_report(fid: str):
    windows = pd.DataFrame(
        [
            {
                "window_index": 0,
                "session_id": 0,
                "t_start": 0.0,
                "t_end": 2.9,
                "anomaly_score": 5.0,
                "threshold": 4.0,
                "is_anomaly": True,
            }
        ]
    )
    return build_flight_report(
        flight_id=fid,
        analysis_status="completed",
        windows=windows,
        threshold=4.0,
        variant="conservative",
        features=FEATURES,
    )


def test_ensure_chat_resets_on_flight_change():
    session: dict = {}
    a = ensure_chat_for_flight(session, "flight-a", "ai_chat")
    a["messages"].append({"role": "user", "content": "about A"})
    b = ensure_chat_for_flight(session, "flight-b", "ai_chat")
    assert b["flight_id"] == "flight-b"
    assert b["messages"] == []


def test_ask_followup_rejects_mismatched_state():
    report = _mini_report("flight-a")
    chat = empty_chat_state("flight-b")
    service = LLMService(client=FakeClient())
    with pytest.raises(ValueError, match="mix flight"):
        ask_followup(
            service,
            report=report,
            chat_state=chat,
            user_message="hello",
        )


def test_ask_followup_binds_history_to_flight():
    report = _mini_report("flight-a")
    chat = empty_chat_state("flight-a")
    diag = DiagnosticReport(
        flight_id="flight-a",
        summary="Summary",
        findings=[],
        overall_limitations=[],
        insufficient_evidence=True,
    )
    service = LLMService(client=FakeClient())
    ans = ask_followup(
        service,
        report=report,
        chat_state=chat,
        user_message="Why flagged?",
        diagnostic=diag,
    )
    assert ans.flight_id == "flight-a"
    assert len(chat["messages"]) == 2
    assert chat["messages"][0]["role"] == "user"
    assert chat["messages"][1]["role"] == "assistant"
