"""State contracts for the chat completion LangGraph agent."""

from typing import Any, Literal, TypedDict

ChatAgentRoute = Literal["simple", "complex"]


class ChatAgentState(TypedDict, total=False):
    """State passed through the LangGraph pipeline."""

    provider: str
    base_url: str
    router_model: str
    model: str
    messages: list[dict[str, str]]
    temperature: float | None
    max_tokens: int | None
    selected_route: ChatAgentRoute
    responder_model: str
    completion: dict[str, Any]
    trace: Any | None


class ChatAgentResult(TypedDict):
    """Public result returned by the chat completion agent."""

    completion: dict[str, Any]
    selected_route: ChatAgentRoute
    responder_model: str
