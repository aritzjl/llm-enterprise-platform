from __future__ import annotations

import logging
import time
from typing import Any

from app.core.config import Settings

try:
    from langfuse import Langfuse
except Exception:  # pragma: no cover - optional dependency at runtime
    Langfuse = None

logger = logging.getLogger(__name__)


def start_observation(
    settings: Settings,
    *,
    provider: str,
    model: str,
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    if not settings.langfuse_enabled:
        return None
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return None
    if Langfuse is None:
        logger.warning("Langfuse SDK is unavailable; continuing without tracing.")
        return None

    try:
        client = Langfuse(
            host=settings.langfuse_host,
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
        )
        trace = client.trace(
            name="session_1_chat_completion",
            input=payload,
            metadata={"provider": provider, "model": model},
        )
        generation = None
        if hasattr(trace, "generation"):
            generation = trace.generation(
                name="llm_generation",
                model=model,
                input=payload,
                metadata={"provider": provider},
            )
        return {
            "client": client,
            "trace": trace,
            "generation": generation,
            "started_at": time.perf_counter(),
        }
    except Exception as exc:  # pragma: no cover - fail-open path
        logger.warning("Langfuse tracing disabled due to init error: %s", exc)
        return None


def finish_observation(
    observation: dict[str, Any] | None,
    *,
    output: Any,
    usage: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    if observation is None:
        return

    latency_ms = int((time.perf_counter() - observation["started_at"]) * 1000)
    metadata = {"latency_ms": latency_ms}
    if error:
        metadata["error"] = error

    try:
        generation = observation.get("generation")
        if generation is not None and hasattr(generation, "end"):
            end_payload: dict[str, Any] = {"output": output, "metadata": metadata}
            if usage is not None:
                end_payload["usage"] = usage
            generation.end(**end_payload)

        trace = observation.get("trace")
        if trace is not None and hasattr(trace, "update"):
            trace.update(output=output, metadata=metadata)

        client = observation.get("client")
        if client is not None and hasattr(client, "flush"):
            client.flush()
    except Exception as exc:  # pragma: no cover - fail-open path
        logger.warning("Langfuse tracing flush failed: %s", exc)
