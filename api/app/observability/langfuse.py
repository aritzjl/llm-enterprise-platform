"""Langfuse helpers for best-effort observability."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from langfuse import Langfuse

from app.core.settings import get_settings
from app.models.chat import ChatRequest

logger = logging.getLogger(__name__)


def _langfuse_enabled() -> bool:
    settings = get_settings()
    return bool(settings.langfuse_public_key and settings.langfuse_secret_key)


@lru_cache(maxsize=1)
def get_langfuse_client() -> Langfuse | None:
    """Return a cached Langfuse client or None when observability is disabled."""
    settings = get_settings()
    if not _langfuse_enabled():
        logger.warning(
            "Langfuse disabled: LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY are missing."
        )
        return None

    try:
        return Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_base_url,
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Failed to initialize Langfuse client: %s", exc)
        return None


def flush_langfuse() -> None:
    """Flush pending telemetry without failing API shutdown."""
    client = get_langfuse_client()
    if client is None:
        return

    try:
        client.flush()
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Failed to flush Langfuse events: %s", exc)


def create_chat_trace(request: ChatRequest) -> Any | None:
    """Create a trace for one chat completion request."""
    client = get_langfuse_client()
    if client is None:
        return None

    settings = get_settings()
    trace_input = {
        "model": request.model,
        "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
    }

    try:
        return client.trace(
            name="chat.completions",
            input=trace_input,
            metadata={"endpoint": "/v1/chat/completions"},
            environment=settings.langfuse_environment,
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Failed to create Langfuse trace: %s", exc)
        return None


def update_chat_trace_success(
    trace: Any | None, completion: dict[str, Any], latency_ms: int
) -> None:
    """Attach successful output metadata to trace."""
    if trace is None:
        return

    try:
        trace.update(
            output=completion,
            metadata={"status": "success", "latency_ms": latency_ms},
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Failed to update Langfuse trace (success): %s", exc)


def update_chat_trace_error(
    trace: Any | None, error_message: str, latency_ms: int
) -> None:
    """Attach error metadata to trace."""
    if trace is None:
        return

    try:
        trace.update(
            metadata={"status": "error", "latency_ms": latency_ms},
            output={"error": error_message},
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Failed to update Langfuse trace (error): %s", exc)


def create_llm_generation(
    trace: Any | None,
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float | None,
    max_tokens: int,
) -> Any | None:
    """Create a child generation for one upstream LLM call."""
    if trace is None:
        return None

    settings = get_settings()
    model_parameters: dict[str, Any] = {"max_tokens": max_tokens}
    if temperature is not None:
        model_parameters["temperature"] = temperature

    try:
        return trace.generation(
            name="llm.generation",
            model=model,
            input=messages,
            model_parameters=model_parameters,
            metadata={"provider": "openai-compatible"},
            environment=settings.langfuse_environment,
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Failed to create Langfuse generation: %s", exc)
        return None


def end_llm_generation_success(
    generation: Any | None,
    *,
    output: dict[str, Any],
    usage_details: dict[str, int] | None,
    latency_ms: int,
) -> None:
    """Mark generation as successful."""
    if generation is None:
        return

    try:
        generation.end(
            output=output,
            usage_details=usage_details,
            metadata={"status": "success", "latency_ms": latency_ms},
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Failed to end Langfuse generation (success): %s", exc)


def end_llm_generation_error(
    generation: Any | None, *, error_message: str, latency_ms: int
) -> None:
    """Mark generation as failed."""
    if generation is None:
        return

    try:
        generation.end(
            level="ERROR",
            status_message=error_message,
            metadata={"status": "error", "latency_ms": latency_ms},
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Failed to end Langfuse generation (error): %s", exc)
