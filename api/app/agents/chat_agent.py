"""Single-node LangGraph chat agent."""

from time import perf_counter
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from app.clients.openai_client import get_openai_client
from app.models.chat import ChatRequest
from app.observability.langfuse import (
    create_llm_generation,
    end_llm_generation_error,
    end_llm_generation_success,
)


class ChatAgentState(TypedDict, total=False):
    """State passed through the LangGraph pipeline."""

    provider: str
    base_url: str
    model: str
    messages: list[dict[str, str]]
    temperature: float | None
    max_tokens: int
    completion: dict[str, Any]
    trace: Any | None


def _extract_usage_details(completion: dict[str, Any]) -> dict[str, int] | None:
    usage = completion.get("usage")
    if not isinstance(usage, dict):
        return None

    details = {
        key: value
        for key, value in usage.items()
        if key in {"prompt_tokens", "completion_tokens", "total_tokens"}
        and isinstance(value, int)
    }
    return details or None


def _call_llm(state: ChatAgentState) -> ChatAgentState:
    start = perf_counter()
    generation = create_llm_generation(
        trace=state.get("trace"),
        provider=state["provider"],
        base_url=state["base_url"],
        model=state["model"],
        messages=state["messages"],
        temperature=state.get("temperature"),
        max_tokens=state["max_tokens"],
    )

    payload: dict[str, Any] = {
        "model": state["model"],
        "messages": state["messages"],
        "max_tokens": state["max_tokens"],
        "stream": False,
    }
    if state.get("temperature") is not None:
        payload["temperature"] = state["temperature"]

    try:
        completion = get_openai_client(base_url=state["base_url"]).chat.completions.create(
            **payload
        )
    except Exception as exc:
        end_llm_generation_error(
            generation=generation,
            error_message=str(exc),
            latency_ms=int((perf_counter() - start) * 1000),
        )
        raise

    completion_json = completion.model_dump(mode="json")
    end_llm_generation_success(
        generation=generation,
        output=completion_json,
        usage_details=_extract_usage_details(completion_json),
        latency_ms=int((perf_counter() - start) * 1000),
    )
    return {"completion": completion_json}


def _build_chat_agent():
    builder = StateGraph(ChatAgentState)
    builder.add_node("call_llm", _call_llm)
    builder.set_entry_point("call_llm")
    builder.add_edge("call_llm", END)
    return builder.compile()


_chat_agent = None


def initialize_chat_agent() -> None:
    """Compile the graph once and cache it for the app lifetime."""
    global _chat_agent
    if _chat_agent is None:
        _chat_agent = _build_chat_agent()


def run_chat_agent(request: ChatRequest, trace: Any | None = None) -> dict[str, Any]:
    """Invoke the precompiled graph for one chat completion request."""
    global _chat_agent
    if _chat_agent is None:
        initialize_chat_agent()

    state: ChatAgentState = {
        "provider": request.provider,
        "base_url": request.base_url,
        "model": request.model,
        "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "trace": trace,
    }
    result = _chat_agent.invoke(state)
    return result["completion"]
