"""Single-node LangGraph chat agent."""

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from app.clients.openai_client import get_openai_client
from app.models.chat import ChatRequest


class ChatAgentState(TypedDict, total=False):
    """State passed through the LangGraph pipeline."""

    model: str
    messages: list[dict[str, str]]
    temperature: float | None
    max_tokens: int
    completion: dict[str, Any]


def _call_llm(state: ChatAgentState) -> ChatAgentState:
    payload: dict[str, Any] = {
        "model": state["model"],
        "messages": state["messages"],
        "max_tokens": state["max_tokens"],
        "stream": False,
    }
    if state.get("temperature") is not None:
        payload["temperature"] = state["temperature"]

    completion = get_openai_client().chat.completions.create(**payload)
    return {"completion": completion.model_dump(mode="json")}


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


def run_chat_agent(request: ChatRequest) -> dict[str, Any]:
    """Invoke the precompiled graph for one chat completion request."""
    global _chat_agent
    if _chat_agent is None:
        initialize_chat_agent()

    state: ChatAgentState = {
        "model": request.model,
        "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
    }
    result = _chat_agent.invoke(state)
    return result["completion"]
