"""HTTP schemas for OpenAI-compatible chat completion."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models.chat import ChatMessage, ChatRequest


class ChatMessageSchema(BaseModel):
    """Input message shape for chat completion requests."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ChatCompletionRequestSchema(BaseModel):
    """Subset of OpenAI chat completion payload supported in session 1."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "provider": "ollama",
                "base_url": "http://localhost:11436/v1",
                "model": "llama3.1:8b",
                "messages": [
                    {"role": "user", "content": "Explica LangGraph en una frase corta."}
                ],
                "temperature": 0.2,
                "max_tokens": 120,
                "stream": False,
            }
        },
    )

    provider: Literal["vllm", "ollama"] = Field(
        description="Proveedor de inferencia para esta llamada (vllm u ollama).",
    )
    base_url: str = Field(
        min_length=1,
        description="Base URL OpenAI-compatible del proveedor para esta llamada.",
        examples=["http://localhost:8001/v1", "http://localhost:11436/v1"],
    )
    model: str = Field(
        min_length=1,
        description="Modelo obligatorio que se usara en esta llamada.",
        examples=["llama3.1:8b"],
    )
    messages: list[ChatMessageSchema] = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int = Field(
        ge=1,
        description="Max tokens obligatorio para esta llamada.",
    )
    stream: Literal[False] = False

    def to_domain(self) -> ChatRequest:
        """Convert transport schema into domain request model."""
        messages = [ChatMessage(role=m.role, content=m.content) for m in self.messages]
        return ChatRequest(
            provider=self.provider,
            base_url=self.base_url.strip(),
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
