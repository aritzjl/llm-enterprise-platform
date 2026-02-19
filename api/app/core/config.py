from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from fastapi import HTTPException


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    vllm_base_url: str
    ollama_base_url: str
    vllm_model: str
    ollama_model: str
    request_timeout_seconds: float
    langfuse_enabled: bool
    langfuse_host: str
    langfuse_public_key: str
    langfuse_secret_key: str


def _read_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@lru_cache
def get_settings() -> Settings:
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "vllm").strip().lower(),
        vllm_base_url=os.getenv("VLLM_BASE_URL", "http://vllm:8000").rstrip("/"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/"),
        vllm_model=os.getenv("VLLM_MODEL", "Qwen/Qwen2-1.5B-Instruct-AWQ"),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen2:1.5b"),
        request_timeout_seconds=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "60")),
        langfuse_enabled=_read_bool_env("LANGFUSE_ENABLED", True),
        langfuse_host=os.getenv("LANGFUSE_HOST", "http://langfuse-web:3000"),
        langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
        langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
    )


def clear_settings_cache() -> None:
    get_settings.cache_clear()


def validate_provider(settings: Settings) -> str:
    if settings.llm_provider not in {"vllm", "ollama"}:
        raise HTTPException(
            status_code=500,
            detail="Invalid LLM_PROVIDER. Expected 'vllm' or 'ollama'.",
        )
    return settings.llm_provider


def default_model_for_provider(settings: Settings, provider: str) -> str:
    if provider == "vllm":
        return settings.vllm_model
    return settings.ollama_model
