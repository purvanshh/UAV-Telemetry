"""Prompt templates for diagnostic reports and flight chat."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from src.evidence_builder import evidence_to_prompt_json


SYSTEM_DIAGNOSTIC = """You are UAV Sentinel's diagnostic assistant.
You interpret structured anomaly-detector evidence for UAV telemetry.
You are NOT a flight-safety authority and must not issue go/no-go decisions.

Rules:
1. The LSTM Autoencoder is the only source of anomaly scores and flags. Never invent or alter them.
2. The deployed model uses exactly 12 conservative IMU/ATT features. BARO altitude/pressure/temperature are excluded from detection — do not attribute LSTM scores to BARO.
3. Separate observed_facts (directly supported by the evidence JSON) from hypotheses.
4. Confidence must be one of: low, medium, high. Never present it as a calibrated probability.
5. If evidence is insufficient for a specific physical fault, set insufficient_evidence=true and avoid asserting a root cause.
6. When events is empty / flagged_windows=0: explain score_max vs threshold, discuss data_quality_warnings as quality context only, and state what additional evidence would be needed. Do not invent events. Windows in top_windows_by_score with below_threshold=true are unflagged context only.
7. Never claim BARO altitude or pressure was detected by the LSTM.
8. Respond with a single JSON object matching the required schema. No markdown fences.
"""

SYSTEM_CHAT = """You are UAV Sentinel's flight-specific diagnostic chat assistant.
Answer only about the provided flight evidence and prior diagnostic report.
Do not invent telemetry. Do not treat prior AI hypotheses as verified facts.
Detector scores and flags are immutable.
Respond with a single JSON object: flight_id, answer, referenced_event_ids, caveats, insufficient_evidence.
No markdown fences.
"""


def diagnostic_response_schema_hint() -> Dict[str, Any]:
    return {
        "flight_id": "string",
        "summary": "string",
        "findings": [
            {
                "event_id": "string",
                "possible_causes": [
                    {
                        "hypothesis": "string",
                        "confidence": "low|medium|high",
                        "supporting_evidence": ["string"],
                        "contradicting_evidence": ["string"],
                    }
                ],
                "observed_facts": ["string"],
                "unknowns": ["string"],
                "recommended_checks": ["string"],
            }
        ],
        "overall_limitations": ["string"],
        "insufficient_evidence": False,
    }


def build_diagnostic_messages(evidence_pack: Dict[str, Any]) -> List[Dict[str, str]]:
    user = (
        "Produce a structured diagnostic JSON for this flight evidence.\n\n"
        f"Required schema shape:\n{json.dumps(diagnostic_response_schema_hint(), indent=2)}\n\n"
        f"Evidence:\n{evidence_to_prompt_json(evidence_pack)}\n"
    )
    return [
        {"role": "system", "content": SYSTEM_DIAGNOSTIC},
        {"role": "user", "content": user},
    ]


def build_chat_messages(
    *,
    evidence_pack: Dict[str, Any],
    diagnostic: Optional[Dict[str, Any]],
    history: List[Dict[str, str]],
    user_message: str,
    max_history: int = 8,
) -> List[Dict[str, str]]:
    context = {
        "evidence": {k: v for k, v in evidence_pack.items() if not str(k).startswith("_")},
        "prior_ai_diagnostic": diagnostic,
        "reminder": (
            "prior_ai_diagnostic contains hypotheses, not verified facts. "
            "Prefer detector evidence fields when answering."
        ),
    }
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_CHAT},
        {
            "role": "user",
            "content": (
                "Flight context (JSON):\n"
                + json.dumps(context, indent=2, ensure_ascii=False)[:20_000]
            ),
        },
    ]
    # Bound history: keep last N user/assistant pairs as plain text turns
    trimmed = history[-max_history:] if history else []
    for turn in trimmed:
        role = turn.get("role", "user")
        if role not in ("user", "assistant"):
            continue
        messages.append({"role": role, "content": str(turn.get("content", ""))[:4000]})
    messages.append({"role": "user", "content": user_message.strip()[:4000]})
    return messages
