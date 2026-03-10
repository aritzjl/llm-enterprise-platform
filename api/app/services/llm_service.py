"""Service layer for model listing and chat completion generation."""

from typing import Any

from openai import APIConnectionError, APIError, APITimeoutError

from app.agents.chat_agent import run_chat_agent
from app.clients.openai_client import get_openai_client
from app.core.exceptions import UpstreamServiceError
from app.models.chat import ChatRequest


def list_models() -> dict[str, Any]:
    """Proxy model listing from the configured OpenAI-compatible provider."""
    client = get_openai_client()
    try:
        response = client.models.list()
    except (APIConnectionError, APITimeoutError, APIError) as exc:
        raise UpstreamServiceError(f"Error listing models from upstream provider: {exc}") from exc
    return response.model_dump(mode="json")


def create_chat_completion(request: ChatRequest) -> dict[str, Any]:
    """Generate one chat completion via LangGraph."""
    try:
        return run_chat_agent(request)
    except (APIConnectionError, APITimeoutError, APIError) as exc:
        raise UpstreamServiceError(
            f"Error generating chat completion from upstream provider: {exc}"
        ) from exc
