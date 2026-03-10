"""Application settings loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the API."""

    app_name: str = "LLM Enterprise Platform API"
    app_description: str = (
        "API base del curso LLMs Locales - Ollama, vLLM y sus Alternativas"
    )
    app_version: str = "0.1.0"
    app_root_path: str = "/curso/api"

    openai_base_url: str = "http://localhost:8001/v1"
    openai_api_key: str = "dummy"
    openai_timeout_seconds: float = 60.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
