from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role del mensaje (system, user, assistant, tool).")
    content: str = Field(..., description="Contenido textual del mensaje.")
    model_config = ConfigDict(extra="allow")


class ChatCompletionRequest(BaseModel):
    model: str | None = Field(
        default=None,
        description="Modelo a usar. Si no se envia, se toma desde el proveedor configurado en .env.",
    )
    messages: list[ChatMessage] = Field(
        ...,
        min_length=1,
        description="Lista de mensajes en formato OpenAI chat.",
    )
    stream: bool = Field(default=False, description="Si es true, devuelve SSE con chunks.")
    temperature: float | None = Field(default=None, description="Temperatura de muestreo.")
    max_tokens: int | None = Field(default=None, description="Maximo de tokens de salida.")
    top_p: float | None = Field(default=None, description="Nucleus sampling.")
    stop: str | list[str] | None = Field(default=None, description="Secuencias de parada.")
    format: str | dict[str, Any] | None = Field(default=None, description="Formato para Ollama.")
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": {
                "model": "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8",
                "messages": [
                    {"role": "system", "content": "Responde en espanol y de forma concisa."},
                    {"role": "user", "content": "Explica RAG en una frase."},
                ],
                "temperature": 0.2,
                "max_tokens": 120,
                "stream": False,
            }
        },
    )
