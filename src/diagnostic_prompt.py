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
The user message includes diagnostic_output, the JSON report already produced for this flight, and detector_evidence, the detector numbers behind it.
Answer the follow-up from diagnostic_output. Cite its event_id values, observed_facts, possible_causes, recommended_checks, and the matching scores or signals in detector_evidence.
Do not reply with generic UAV advice that could apply to any flight. Name the events and signals this flight actually has.
Do not invent telemetry. Do not treat hypotheses as verified faults. Detector scores and flags are immutable.
If the user asks for preventive measures or checks, expand the recommended_checks already listed for the relevant events and say which event_id and signal or score each check comes from.
Set insufficient_evidence to true only when diagnostic_output has no findings and detector_evidence has no events.
Respond with a single JSON object: flight_id, answer (non-empty string), referenced_event_ids (array of strings), caveats (array of strings), insufficient_evidence (boolean).
No markdown fences. Do not return caveats or referenced_event_ids as a single string.
"""

# Diagnostic JSON is never truncated. Evidence is shrunk only after this budget.
CHAT_CONTEXT_CHAR_BUDGET = 48_000


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


def _json_len(obj: Any) -> int:
    return len(json.dumps(obj, ensure_ascii=False))


def _shrink_evidence(evidence_pack: Dict[str, Any], budget: int) -> Dict[str, Any]:
    """Drop bulky evidence fields until the pack fits. Never touch the diagnostic."""
    clean: Dict[str, Any] = {
        k: v for k, v in evidence_pack.items() if not str(k).startswith("_")
    }
    if _json_len(clean) <= budget:
        return clean
    for key in ("instructions_for_model", "detector_limitations"):
        clean.pop(key, None)
        if _json_len(clean) <= budget:
            return clean
    events = clean.get("events")
    if isinstance(events, list):
        for field in ("imu_att_summary", "baseline_windows", "evidence"):
            if _json_len(clean) <= budget:
                return clean
            for event in events:
                if isinstance(event, dict):
                    event.pop(field, None)
    windows = clean.get("top_windows_by_score")
    if isinstance(windows, list) and _json_len(clean) > budget:
        for window in windows:
            if isinstance(window, dict):
                window.pop("top_feature_recon_errors", None)
        while windows and _json_len(clean) > budget:
            windows.pop()
    if isinstance(events, list):
        while len(events) > 1 and _json_len(clean) > budget:
            events.pop()
            clean["events_omitted"] = int(clean.get("events_omitted") or 0) + 1
    if _json_len(clean) > budget:
        clean = {
            "flight_id": clean.get("flight_id"),
            "analysis_status": clean.get("analysis_status"),
            "detector": clean.get("detector"),
            "summary": clean.get("summary"),
            "score_context": clean.get("score_context"),
            "events": events if isinstance(events, list) else [],
            "context_truncated": True,
        }
    return clean


def build_chat_messages(
    *,
    evidence_pack: Dict[str, Any],
    diagnostic: Optional[Dict[str, Any]],
    history: List[Dict[str, str]],
    user_message: str,
    max_history: int = 8,
) -> List[Dict[str, str]]:
    diagnostic_obj = diagnostic if isinstance(diagnostic, dict) else {}
    diagnostic_json = json.dumps(diagnostic_obj, indent=2, ensure_ascii=False)
    evidence_budget = max(4_000, CHAT_CONTEXT_CHAR_BUDGET - len(diagnostic_json) - 800)
    evidence_json = json.dumps(
        _shrink_evidence(evidence_pack, evidence_budget),
        indent=2,
        ensure_ascii=False,
    )
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_CHAT},
        {
            "role": "user",
            "content": (
                "Use diagnostic_output as the context for this follow-up. "
                "Cite its findings. Use detector_evidence for the exact scores and signals. "
                "Hypotheses in diagnostic_output are not verified faults. "
                "If an earlier assistant reply is generic, ignore it and answer from diagnostic_output.\n\n"
                "diagnostic_output:\n"
                f"{diagnostic_json}\n\n"
                "detector_evidence:\n"
                f"{evidence_json}\n"
            ),
        },
    ]
    trimmed = history[-max_history:] if history else []
    for turn in trimmed:
        role = turn.get("role", "user")
        if role not in ("user", "assistant"):
            continue
        messages.append({"role": role, "content": str(turn.get("content", ""))[:4000]})
    messages.append({"role": "user", "content": user_message.strip()[:4000]})
    return messages
