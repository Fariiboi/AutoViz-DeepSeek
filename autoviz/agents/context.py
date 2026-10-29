"""Per-request LLM configuration override.

When a user provides deepseek_api_key and/or ai_model via WebSocket,
they override the default settings for that request. Uses contextvars
for async-safe per-request state.
"""

import contextvars
from typing import NamedTuple

from autoviz.config import settings

_llm_override: contextvars.ContextVar["LLMOverride | None"] = contextvars.ContextVar(
    "llm_override", default=None
)


class LLMOverride(NamedTuple):
    """Optional per-request overrides for DeepSeek API key and model."""

    api_key: str | None
    model: str | None


def set_llm_override(api_key: str | None = None, model: str | None = None) -> None:
    """Set LLM overrides for the current async context."""
    if api_key or model:
        _llm_override.set(LLMOverride(api_key=api_key, model=model))
    else:
        _llm_override.set(None)


def clear_llm_override() -> None:
    """Clear LLM overrides."""
    _llm_override.set(None)


def get_effective_api_key() -> str:
    """Return API key: override if set, else settings."""
    override = _llm_override.get()
    if override and override.api_key and override.api_key.strip():
        return override.api_key.strip()
    return settings.DEEPSEEK_API_KEY or ""


def get_effective_model() -> str:
    """Return model: override if set, else settings."""
    override = _llm_override.get()
    if override and override.model and override.model.strip():
        return override.model.strip()
    return settings.AI_MODEL or "deepseek-chat"


def has_llm_override() -> bool:
    """Return True if per-request override is set."""
    return _llm_override.get() is not None


# ── Language context ──────────────────────────────────────────────────────────

_language_context: contextvars.ContextVar[str] = contextvars.ContextVar(
    "language_context", default="en"
)


def set_language_context(language: str) -> None:
    """Set language for the current async context. Accepts 'en' or 'zh'."""
    _language_context.set(language if language in ("en", "zh") else "en")


def clear_language_context() -> None:
    """Reset language to default ('en') for the current async context."""
    _language_context.set("en")


def get_language_context() -> str:
    """Return current language: 'en' or 'zh'."""
    return _language_context.get()


def get_language_instruction() -> str:
    """Return a language instruction string to append to agent system prompts."""
    lang = _language_context.get()
    if lang == "zh":
        return "\n\nPlease respond in Simplified Chinese."
    return "\n\nPlease respond in English."
