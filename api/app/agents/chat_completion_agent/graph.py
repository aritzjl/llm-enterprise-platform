"""Graph assembly and runtime for the chat completion agent."""

from __future__ import annotations

from typing import Any, cast

from langgraph.graph import END, StateGraph

from app.models.chat import ChatRequest

from .nodes import (
    call_primary_model_node,
    call_router_model_node,
    classify_route_node,
    route_selector,
)
from .state import ChatAgentResult, ChatAgentState


def _build_chat_agent():
    builder = StateGraph(ChatAgentState)
    builder.add_node("classify_route", classify_route_node)
    builder.add_node("call_router_model", call_router_model_node)
    builder.add_node("call_primary_model", call_primary_model_node)

    builder.set_entry_point("classify_route")
    builder.add_conditional_edges(
        "classify_route",
        route_selector,
        {
            "simple": "call_router_model",
            "complex": "call_primary_model",
        },
    )
    builder.add_edge("call_router_model", END)
    builder.add_edge("call_primary_model", END)
    return builder.compile()


_chat_agent = None


def initialize_chat_agent() -> None:
    """Compile the graph once and cache it for the app lifetime."""
    global _chat_agent
    if _chat_agent is None:
        _chat_agent = _build_chat_agent()


def run_chat_agent(request: ChatRequest, trace: Any | None = None) -> ChatAgentResult:
    """Invoke the precompiled graph for one chat completion request."""
    global _chat_agent
    if _chat_agent is None:
        initialize_chat_agent()

    state: ChatAgentState = {
        "provider": request.provider,
        "base_url": request.base_url,
        "router_model": request.router_model,
        "model": request.model,
        "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "trace": trace,
    }
    result = cast(ChatAgentState, _chat_agent.invoke(state))
    selected_route = route_selector(result)
    responder_model = result.get(
        "responder_model",
        request.router_model if selected_route == "simple" else request.model,
    )

    return {
        "completion": result["completion"],
        "selected_route": selected_route,
        "responder_model": responder_model,
    }
