"""OpenAI GPT-4o mini client for diagnostics and chat.

API keys are read from the environment only. Routine tests inject a mock
client and must not perform paid calls.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol

from dotenv import load_dotenv

from src.diagnostic_prompt import build_chat_messages, build_diagnostic_messages
from src.diagnostic_schema import (
    ChatAnswer,
    DiagnosticReport,
    insufficient_evidence_report,
    validate_chat_answer,
    validate_diagnostic,
)
from src.evidence_builder import build_evidence_pack
from src.report_schema import FlightReport

load_dotenv()

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TIMEOUT_S = 60.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_TEMPERATURE = 0.2


class LLMError(RuntimeError):
    """User-safe LLM failure (no secrets in message)."""


class LLMConfigError(LLMError):
    pass


class LLMResponseError(LLMError):
    pass


class ChatCompletionsClient(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


@dataclass
class LLMUsage:
    latency_ms: float
    model: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


def get_api_key() -> Optional[str]:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    return key or None


def get_model_name() -> str:
    return os.environ.get("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def api_configured() -> bool:
    return get_api_key() is not None


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text


def _parse_json_content(content: str) -> Dict[str, Any]:
    raw = _strip_json_fence(content)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMResponseError(
            "Model returned non-JSON content that could not be parsed."
        ) from exc
    if not isinstance(data, dict):
        raise LLMResponseError("Model JSON root must be an object.")
    return data


def _extract_message_content(response: Any) -> str:
    try:
        return response.choices[0].message.content or ""
    except Exception as exc:
        raise LLMResponseError("Unexpected API response shape.") from exc


def _usage_from_response(response: Any, model: str, latency_ms: float) -> LLMUsage:
    prompt_tokens = completion_tokens = None
    usage = getattr(response, "usage", None)
    if usage is not None:
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
    return LLMUsage(
        latency_ms=latency_ms,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


class OpenAIChatClient:
    """Thin wrapper around openai.OpenAI chat.completions."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ):
        key = api_key if api_key is not None else get_api_key()
        if not key:
            raise LLMConfigError(
                "OPENAI_API_KEY is not set. Add it to your local .env file "
                "(see .env.example). Detector results remain available without AI."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMConfigError(
                "The 'openai' package is not installed. Run: pip install openai"
            ) from exc
        self._client = OpenAI(api_key=key, timeout=timeout_s)

    def create(self, **kwargs: Any) -> Any:
        return self._client.chat.completions.create(**kwargs)


def _call_with_retries(
    create_fn: Callable[..., Any],
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    **kwargs: Any,
) -> Any:
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            return create_fn(**kwargs)
        except LLMError:
            raise
        except Exception as exc:
            last_exc = exc
            name = type(exc).__name__.lower()
            msg = str(exc).lower()
            # Don't retry auth errors
            if "auth" in name or "unauthorized" in msg or "invalid_api_key" in msg:
                raise LLMConfigError(
                    "OpenAI authentication failed. Check OPENAI_API_KEY."
                ) from exc
            if attempt >= max_retries:
                break
            # Brief backoff for rate limits / transient failures
            time.sleep(0.6 * (attempt + 1))
    raise LLMError(
        "OpenAI request failed after retries. Try again later. "
        "Detector reports and exports are unaffected."
    ) from last_exc


class LLMService:
    def __init__(
        self,
        client: Optional[ChatCompletionsClient] = None,
        *,
        model: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        self.model = model or get_model_name()
        self.temperature = temperature
        self.max_retries = max_retries
        self._client = client
        self.last_usage: Optional[LLMUsage] = None
        self.last_evidence_pack: Optional[Dict[str, Any]] = None

    def _ensure_client(self) -> ChatCompletionsClient:
        if self._client is None:
            self._client = OpenAIChatClient()
        return self._client

    def _complete(self, messages: List[Dict[str, str]]) -> str:
        client = self._ensure_client()
        t0 = time.perf_counter()
        response = _call_with_retries(
            client.create,
            max_retries=self.max_retries,
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self.last_usage = _usage_from_response(response, self.model, latency_ms)
        return _extract_message_content(response)

    def generate_diagnostic(
        self,
        report: FlightReport,
        *,
        windows=None,
        aligned=None,
        selected_event_ids: Optional[List[str]] = None,
    ) -> DiagnosticReport:
        # Local short-circuit only when analysis itself failed / had no windows.
        # Zero-event / no_anomaly flights still go to the model with score_context.
        if report.analysis_status in ("insufficient_data", "error"):
            return insufficient_evidence_report(
                report.flight_id,
                reason=f"Analysis status is '{report.analysis_status}'.",
                model_name=self.model,
            )

        pack = build_evidence_pack(
            report,
            windows=windows,
            aligned=aligned,
            selected_event_ids=selected_event_ids,
        )
        self.last_evidence_pack = pack
        messages = build_diagnostic_messages(pack)
        content = self._complete(messages)
        data = _parse_json_content(content)
        data.setdefault("flight_id", report.flight_id)
        data["model_name"] = self.model
        try:
            return validate_diagnostic(data)
        except Exception as exc:
            raise LLMResponseError(
                "Model JSON did not match the diagnostic schema."
            ) from exc

    def chat(
        self,
        *,
        report: FlightReport,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
        diagnostic: Optional[DiagnosticReport] = None,
        windows=None,
        aligned=None,
        selected_event_ids: Optional[List[str]] = None,
    ) -> ChatAnswer:
        if not user_message or not str(user_message).strip():
            raise LLMError("Chat message is empty.")
        pack = build_evidence_pack(
            report,
            windows=windows,
            aligned=aligned,
            selected_event_ids=selected_event_ids,
        )
        self.last_evidence_pack = pack
        diag_dict = diagnostic.to_json_dict() if diagnostic else None
        messages = build_chat_messages(
            evidence_pack=pack,
            diagnostic=diag_dict,
            history=list(history or []),
            user_message=user_message,
        )
        content = self._complete(messages)
        data = _parse_json_content(content)
        data.setdefault("flight_id", report.flight_id)
        try:
            return validate_chat_answer(data)
        except Exception as exc:
            raise LLMResponseError(
                "Model JSON did not match the chat answer schema."
            ) from exc


def attach_ai_to_report(
    report: FlightReport, diagnostic: DiagnosticReport
) -> FlightReport:
    """Return a copy of the flight report with ai_analysis populated."""
    return report.model_copy(
        update={"ai_analysis": diagnostic.to_json_dict()},
        deep=True,
    )
