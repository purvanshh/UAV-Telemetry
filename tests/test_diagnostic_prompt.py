"""Language invariants for the pilot-facing AI layer.

The diagnostic + chat prompts must stay in plain language for a pilot with no
data-science background, keep the fixed vocabulary, and keep every guardrail
(immutable detector numbers, no fault claims, JSON-only output).
"""

from __future__ import annotations

import re

from src.diagnostic_prompt import (
    PLAIN_LANGUAGE_GUIDE,
    SYSTEM_CHAT,
    SYSTEM_DIAGNOSTIC,
    build_chat_messages,
    build_diagnostic_messages,
    diagnostic_response_schema_hint,
)
from src.diagnostic_schema import insufficient_evidence_report


def test_plain_language_guide_present_in_both_prompts():
    assert PLAIN_LANGUAGE_GUIDE in SYSTEM_DIAGNOSTIC
    assert PLAIN_LANGUAGE_GUIDE in SYSTEM_CHAT


def test_guide_fixes_the_vocabulary():
    for term in (
        "unusualness score",
        "alert level",
        "flagged moment",
        "unusual stretch",
        "motion sensors",
        "tilt and heading data",
        "tilt tracking",
        "barometer",
    ):
        assert term in PLAIN_LANGUAGE_GUIDE


def test_guide_bans_internal_jargon():
    assert "Never use these terms" in PLAIN_LANGUAGE_GUIDE
    for banned in ("reconstruction", "MSE", "z-score", "autoencoder", "LSTM"):
        assert banned in PLAIN_LANGUAGE_GUIDE  # listed as forbidden


def test_guardrails_survive():
    assert "go/no-go" in SYSTEM_DIAGNOSTIC
    assert "flight-safety authority" in SYSTEM_DIAGNOSTIC.lower()
    for prompt in (SYSTEM_DIAGNOSTIC, SYSTEM_CHAT):
        low = prompt.lower()
        assert "never invent" in low, "numbers must stay immutable"
        assert "json object" in low, "outputs must stay machine-parsable"
        assert "never present it as a confirmed fault" in low or "never as confirmed" in low


def test_schema_hint_fields_unchanged():
    hint = diagnostic_response_schema_hint()
    assert set(hint) == {
        "flight_id",
        "summary",
        "findings",
        "overall_limitations",
        "insufficient_evidence",
    }


def test_user_messages_state_pilot_audience():
    pack = {"flight_id": "f1", "events": [], "score_context": {}, "summary": {}}

    diag_msgs = build_diagnostic_messages(pack)
    assert diag_msgs[0]["content"] is SYSTEM_DIAGNOSTIC
    assert "pilot" in diag_msgs[1]["content"].lower()

    chat_msgs = build_chat_messages(
        evidence_pack=pack, diagnostic=None, history=[], user_message="Why did it flag?"
    )
    assert chat_msgs[0]["content"] is SYSTEM_CHAT
    assert "pilot" in chat_msgs[1]["content"].lower()


def test_local_insufficient_report_is_plain():
    r = insufficient_evidence_report("f1", reason="The flight data was too short.")
    text = (r.summary + " " + " ".join(r.overall_limitations)).lower()
    for banned in ("lstm", "baro", "reconstruction", "imu", "att"):
        assert not re.search(rf"\b{banned}\b", text)
    assert r.insufficient_evidence is True
    assert r.findings == []
