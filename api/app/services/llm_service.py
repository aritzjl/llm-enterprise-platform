"""Service layer for model listing and chat completion generation."""

from time import perf_counter
from typing import Any

from openai import APIConnectionError, APIError, APITimeoutError

from app.agents.chat_agent import run_chat_agent
from app.clients.openai_client import get_openai_client
from app.core.exceptions import UpstreamServiceError
from app.models.chat import ChatRequest
from app.observability.langfuse import (
    create_chat_trace,
    update_chat_trace_error,
    update_chat_trace_success,
)


def list_models() -> dict[str, Any]:
    """Proxy model listing from the configured OpenAI-compatible provider."""
    client = get_openai_client()
    try:
        response = client.models.list()
    except (APIConnectionError, APITimeoutError, APIError) as exc:
        raise UpstreamServiceError(f"Error listing models from upstream provider: {exc}") from exc
    return response.model_dump(mode="json")


def create_chat_completion(request: ChatRequest) -> dict[str, Any]:
    """Generate one chat completion via LangGraph."""
    trace = create_chat_trace(request)
    start = perf_counter()
    try:
        completion = run_chat_agent(request, trace=trace)
    except (APIConnectionError, APITimeoutError, APIError) as exc:
        update_chat_trace_error(
            trace=trace, error_message=str(exc), latency_ms=int((perf_counter() - start) * 1000)
        )
        raise UpstreamServiceError(
            f"Error generating chat completion from upstream provider: {exc}"
        ) from exc
    except Exception as exc:
        update_chat_trace_error(
            trace=trace, error_message=str(exc), latency_ms=int((perf_counter() - start) * 1000)
        )
        raise

    update_chat_trace_success(
        trace=trace,
        completion=completion,
        latency_ms=int((perf_counter() - start) * 1000),
    )
    return completion
