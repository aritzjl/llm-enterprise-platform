"""OpenAI client factory configured via environment variables."""

from functools import lru_cache

from openai import OpenAI

from app.core.settings import get_settings


@lru_cache(maxsize=16)
def _build_openai_client(base_url: str, api_key: str, timeout: float) -> OpenAI:
    """Build and cache clients by target base URL."""
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
    )


def get_openai_client(base_url: str | None = None) -> OpenAI:
    """Return a cached OpenAI-compatible client for the requested base URL."""
    settings = get_settings()
    selected_base_url = (base_url or settings.openai_base_url).strip()
    if not selected_base_url:
        selected_base_url = settings.openai_base_url

    return _build_openai_client(
        base_url=selected_base_url,
        api_key=settings.openai_api_key,
        timeout=settings.openai_timeout_seconds,
    )
