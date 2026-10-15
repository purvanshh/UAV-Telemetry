"""Diagnostic / chat schema validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.diagnostic_schema import (
    DiagnosticReport,
    insufficient_evidence_report,
    validate_chat_answer,
    validate_diagnostic,
)


def test_valid_diagnostic():
    report = validate_diagnostic(
        {
            "flight_id": "f1",
            "summary": "Unusual gyro reconstruction error near t=12s.",
            "findings": [
                {
                    "event_id": "event-001",
                    "possible_causes": [
                        {
                            "hypothesis": "Transient gyroscope disturbance",
                            "confidence": "low",
                            "supporting_evidence": ["peak feat_err_gyro_x"],
                            "contradicting_evidence": [],
                        }
                    ],
                    "observed_facts": ["peak_score exceeded threshold"],
                    "unknowns": ["No motor current telemetry"],
                    "recommended_checks": ["Compare gyro axes"],
                }
            ],
            "overall_limitations": ["12-feature conservative model"],
            "insufficient_evidence": False,
        }
    )
    assert report.findings[0].possible_causes[0].confidence == "low"


def test_rejects_bad_confidence():
    with pytest.raises(ValidationError):
        validate_diagnostic(
            {
                "flight_id": "f1",
                "summary": "x",
                "findings": [
                    {
                        "event_id": "event-001",
                        "possible_causes": [
                            {
                                "hypothesis": "h",
                                "confidence": "0.91",
                                "supporting_evidence": [],
                                "contradicting_evidence": [],
                            }
                        ],
                        "observed_facts": [],
                        "unknowns": [],
                        "recommended_checks": [],
                    }
                ],
                "overall_limitations": [],
                "insufficient_evidence": False,
            }
        )


def test_insufficient_helper():
    r = insufficient_evidence_report("f1", reason="No events.")
    assert r.insufficient_evidence is True
    assert r.findings == []


def test_chat_answer():
    ans = validate_chat_answer(
        {
            "flight_id": "f1",
            "answer": "The flag reflects IMU/ATT reconstruction only.",
            "referenced_event_ids": ["event-001"],
            "caveats": ["Not a root-cause proof"],
            "insufficient_evidence": False,
        }
    )
    assert "IMU/ATT" in ans.answer


def test_chat_answer_accepts_string_caveats():
    ans = validate_chat_answer(
        {
            "flight_id": "f1",
            "answer": "Check the gyro mounting.",
            "referenced_event_ids": "event-001",
            "caveats": "This is a hypothesis, not a confirmed fault.",
            "insufficient_evidence": False,
            "extra_field": "ignored",
        }
    )
    assert ans.caveats == ["This is a hypothesis, not a confirmed fault."]
    assert ans.referenced_event_ids == ["event-001"]
    assert ans.to_json_dict()["answer"] == "Check the gyro mounting."


def test_empty_summary_rejected():
    with pytest.raises(ValidationError):
        DiagnosticReport(
            flight_id="f1",
            summary="   ",
            findings=[],
            overall_limitations=[],
            insufficient_evidence=True,
        )
