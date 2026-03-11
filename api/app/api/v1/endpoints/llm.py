"""LLM endpoints with OpenAI-compatible contracts."""

from typing import Any
from typing import Literal

from fastapi import APIRouter, Query

from app.schemas.chat import ChatCompletionRequestSchema
from app.services import llm_service

router = APIRouter(prefix="/v1", tags=["LLM"])


@router.get("/models")
def list_models(
    base_url: str = Query(
        ...,
        min_length=1,
        description="Base URL OpenAI-compatible del proveedor para esta llamada.",
    ),
    provider: Literal["vllm", "ollama"] | None = Query(
        None,
        description="Proveedor opcional para etiquetar la consulta de modelos.",
    ),
) -> dict[str, Any]:
    """List available models from a request-selected OpenAI-compatible backend."""
    return llm_service.list_models(base_url=base_url.strip(), provider=provider)


@router.post("/chat/completions")
def create_chat_completion(payload: ChatCompletionRequestSchema) -> dict[str, Any]:
    """Generate a chat completion with a LangGraph router-first agent."""
    request = payload.to_domain()
    return llm_service.create_chat_completion(request)
