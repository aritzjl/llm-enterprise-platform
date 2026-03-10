"""Domain models for chat use cases."""

from dataclasses import dataclass


@dataclass(slots=True)
class ChatMessage:
    """Single chat message in domain format."""

    role: str
    content: str


@dataclass(slots=True)
class ChatRequest:
    """Domain request for a single non-stream chat completion."""

    provider: str
    base_url: str
    model: str
    messages: list[ChatMessage]
    max_tokens: int | None = None
    temperature: float | None = None
