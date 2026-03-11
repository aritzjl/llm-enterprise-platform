"""Service layer for model listing and chat completion generation."""

from time import perf_counter
from typing import Any

from openai import APIConnectionError, APIError, APITimeoutError

from app.agents.chat_completion_agent import run_chat_agent
from app.clients.openai_client import get_openai_client
from app.core.exceptions import UpstreamServiceError
from app.models.chat import ChatRequest
from app.observability.langfuse import (
    create_chat_trace,
    update_chat_trace_error,
    update_chat_trace_success,
)


def list_models(base_url: str, provider: str | None = None) -> dict[str, Any]:
    """Proxy model listing from a request-selected OpenAI-compatible provider."""
    client = get_openai_client(base_url=base_url)
    try:
        response = client.models.list()
    except (APIConnectionError, APITimeoutError, APIError) as exc:
        provider_label = provider or "unknown"
        raise UpstreamServiceError(
            f"Error listing models from upstream provider ({provider_label}, {base_url}): {exc}"
        ) from exc
    return response.model_dump(mode="json")


def create_chat_completion(request: ChatRequest) -> dict[str, Any]:
    """Generate one chat completion via LangGraph."""
    trace = create_chat_trace(request)
    start = perf_counter()
    selected_route: str | None = None
    responder_model: str | None = None
    try:
        agent_result = run_chat_agent(request, trace=trace)
        completion = agent_result["completion"]
        selected_route = agent_result["selected_route"]
        responder_model = agent_result["responder_model"]
    except (APIConnectionError, APITimeoutError, APIError) as exc:
        update_chat_trace_error(
            trace=trace,
            error_message=str(exc),
            latency_ms=int((perf_counter() - start) * 1000),
            provider=request.provider,
            base_url=request.base_url,
            router_model=request.router_model,
            selected_route=selected_route,
            responder_model=responder_model,
        )
        raise UpstreamServiceError(
            f"Error generating chat completion from upstream provider: {exc}"
        ) from exc
    except Exception as exc:
        update_chat_trace_error(
            trace=trace,
            error_message=str(exc),
            latency_ms=int((perf_counter() - start) * 1000),
            provider=request.provider,
            base_url=request.base_url,
            router_model=request.router_model,
            selected_route=selected_route,
            responder_model=responder_model,
        )
        raise

    update_chat_trace_success(
        trace=trace,
        completion=completion,
        latency_ms=int((perf_counter() - start) * 1000),
        provider=request.provider,
        base_url=request.base_url,
        router_model=request.router_model,
        selected_route=selected_route,
        responder_model=responder_model,
    )
    return completion
