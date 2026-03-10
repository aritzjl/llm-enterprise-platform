"""OpenAI client factory configured via environment variables."""

from functools import lru_cache

from openai import OpenAI

from app.core.settings import get_settings


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    """Return a cached OpenAI-compatible client."""
    settings = get_settings()
    return OpenAI(
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        timeout=settings.openai_timeout_seconds,
    )
