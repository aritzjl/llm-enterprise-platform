"""LLM endpoints with OpenAI-compatible contracts."""

from typing import Any

from fastapi import APIRouter

from app.schemas.chat import ChatCompletionRequestSchema
from app.services import llm_service

router = APIRouter(prefix="/v1", tags=["LLM"])


@router.get("/models")
def list_models() -> dict[str, Any]:
    """List available models from the configured OpenAI-compatible backend."""
    return llm_service.list_models()


@router.post("/chat/completions")
def create_chat_completion(payload: ChatCompletionRequestSchema) -> dict[str, Any]:
    """Generate a chat completion with a LangGraph-backed single-node agent."""
    request = payload.to_domain()
    return llm_service.create_chat_completion(request)
