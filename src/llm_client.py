"""OpenAI client used by Agentic Analysis. Keys come from the environment only; tests inject a fake client."""

from __future__ import annotations

import os
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

# Measured on the ARJUNA dataset: gpt-4o-mini skips verification and misorders causes; gpt-5.4-mini follows the
# investigation protocol. Override with UAV_AGENT_MODEL (e.g. gpt-5.5 for the deepest investigation).
DEFAULT_MODEL = "gpt-5.4-mini"


class LLMError(RuntimeError):
    """User-safe LLM failure (no secrets in the message)."""


class LLMConfigError(LLMError):
    pass


def get_api_key() -> Optional[str]:
    return os.environ.get("OPENAI_API_KEY", "").strip() or None


def api_configured() -> bool:
    return get_api_key() is not None


def get_model_name() -> str:
    return os.environ.get("UAV_AGENT_MODEL", "").strip() or DEFAULT_MODEL


class OpenAIChatClient:
    def __init__(self, *, api_key: Optional[str] = None, timeout_s: float = 90.0):
        key = api_key if api_key is not None else get_api_key()
        if not key:
            raise LLMConfigError("OPENAI_API_KEY is not set on the server.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMConfigError("The 'openai' package is not installed.") from exc
        self._client = OpenAI(api_key=key, timeout=timeout_s)

    def create(self, **kwargs: Any) -> Any:
        return self._client.chat.completions.create(**kwargs)
