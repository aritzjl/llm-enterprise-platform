"""
LLM Enterprise Platform - API Base
===================================
Servicio FastAPI base para el curso de LLMs Locales.
Este servicio se extiende con chat OpenAI-compatible para la sesion 1.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

try:
    from langfuse import Langfuse
except Exception:  # pragma: no cover - optional dependency at runtime
    Langfuse = None

logger = logging.getLogger(__name__)


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


app = FastAPI(
    title="LLM Enterprise Platform - API",
    description="API base del curso LLMs Locales - Ollama, vLLM y sus Alternativas",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    root_path="/curso/api",
)


def _default_model_for_provider(settings: Settings, provider: str) -> str:
    if provider == "vllm":
        return settings.vllm_model
    return settings.ollama_model


def _validate_provider(settings: Settings) -> str:
    if settings.llm_provider not in {"vllm", "ollama"}:
        raise HTTPException(
            status_code=500,
            detail="Invalid LLM_PROVIDER. Expected 'vllm' or 'ollama'.",
        )
    return settings.llm_provider


def _validate_chat_payload(payload: dict[str, Any]) -> None:
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="Field 'messages' must be a non-empty list.")


def _normalize_openai_payload(payload: dict[str, Any], model: str, stream: bool) -> dict[str, Any]:
    request_payload = dict(payload)
    request_payload["model"] = model
    request_payload["stream"] = stream
    return request_payload


def _build_ollama_payload(payload: dict[str, Any], model: str, stream: bool) -> dict[str, Any]:
    ollama_payload: dict[str, Any] = {
        "model": model,
        "messages": payload.get("messages", []),
        "stream": stream,
    }

    options: dict[str, Any] = {}
    if payload.get("temperature") is not None:
        options["temperature"] = payload["temperature"]
    if payload.get("max_tokens") is not None:
        options["num_predict"] = payload["max_tokens"]
    if payload.get("top_p") is not None:
        options["top_p"] = payload["top_p"]
    if payload.get("stop") is not None:
        options["stop"] = payload["stop"]

    if options:
        ollama_payload["options"] = options

    if payload.get("format") is not None:
        ollama_payload["format"] = payload["format"]

    return ollama_payload


def _build_openai_chat_response(
    *,
    model: str,
    content: str,
    finish_reason: str = "stop",
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    response: dict[str, Any] = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage is not None:
        response["usage"] = usage
    return response


def _build_openai_chunk(
    *,
    completion_id: str,
    model: str,
    delta: dict[str, Any],
    finish_reason: str | None,
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def _extract_content_from_openai_chunk(chunk: dict[str, Any]) -> str:
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    delta = choices[0].get("delta", {})
    if not isinstance(delta, dict):
        return ""
    content = delta.get("content")
    if isinstance(content, str):
        return content
    return ""


def _usage_from_ollama(response: dict[str, Any]) -> dict[str, int] | None:
    prompt_tokens = response.get("prompt_eval_count")
    completion_tokens = response.get("eval_count")
    if prompt_tokens is None and completion_tokens is None:
        return None
    prompt_value = int(prompt_tokens or 0)
    completion_value = int(completion_tokens or 0)
    return {
        "prompt_tokens": prompt_value,
        "completion_tokens": completion_value,
        "total_tokens": prompt_value + completion_value,
    }


def _http_exception_from_error(exc: Exception) -> HTTPException:
    if isinstance(exc, httpx.TimeoutException):
        return HTTPException(status_code=504, detail="LLM provider request timed out.")
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if 400 <= status < 500:
            return HTTPException(status_code=status, detail=f"Provider returned {status}.")
        return HTTPException(status_code=502, detail=f"Provider returned {status}.")
    if isinstance(exc, httpx.RequestError):
        return HTTPException(status_code=502, detail="Unable to reach configured LLM provider.")
    return HTTPException(status_code=502, detail="Unexpected provider error.")


def _start_langfuse_observation(
    settings: Settings,
    *,
    provider: str,
    model: str,
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    if not settings.langfuse_enabled:
        return None
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return None
    if Langfuse is None:
        logger.warning("Langfuse SDK is unavailable; continuing without tracing.")
        return None

    try:
        client = Langfuse(
            host=settings.langfuse_host,
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
        )
        trace = client.trace(
            name="session_1_chat_completion",
            input=payload,
            metadata={"provider": provider, "model": model},
        )
        generation = None
        if hasattr(trace, "generation"):
            generation = trace.generation(
                name="llm_generation",
                model=model,
                input=payload,
                metadata={"provider": provider},
            )
        return {
            "client": client,
            "trace": trace,
            "generation": generation,
            "started_at": time.perf_counter(),
        }
    except Exception as exc:  # pragma: no cover - fail-open path
        logger.warning("Langfuse tracing disabled due to init error: %s", exc)
        return None


def _finish_langfuse_observation(
    observation: dict[str, Any] | None,
    *,
    output: Any,
    usage: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    if observation is None:
        return

    latency_ms = int((time.perf_counter() - observation["started_at"]) * 1000)
    metadata = {"latency_ms": latency_ms}
    if error:
        metadata["error"] = error

    try:
        generation = observation.get("generation")
        if generation is not None and hasattr(generation, "end"):
            end_payload: dict[str, Any] = {"output": output, "metadata": metadata}
            if usage is not None:
                end_payload["usage"] = usage
            generation.end(**end_payload)

        trace = observation.get("trace")
        if trace is not None and hasattr(trace, "update"):
            trace.update(output=output, metadata=metadata)

        client = observation.get("client")
        if client is not None and hasattr(client, "flush"):
            client.flush()
    except Exception as exc:  # pragma: no cover - fail-open path
        logger.warning("Langfuse tracing flush failed: %s", exc)


async def _call_vllm_non_stream(payload: dict[str, Any], settings: Settings) -> dict[str, Any]:
    timeout = httpx.Timeout(settings.request_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(f"{settings.vllm_base_url}/v1/chat/completions", json=payload)
        response.raise_for_status()
        return response.json()


async def _call_ollama_non_stream(
    payload: dict[str, Any],
    settings: Settings,
    model: str,
) -> dict[str, Any]:
    timeout = httpx.Timeout(settings.request_timeout_seconds)
    ollama_payload = _build_ollama_payload(payload, model, stream=False)

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(f"{settings.ollama_base_url}/api/chat", json=ollama_payload)
        response.raise_for_status()
        raw = response.json()

    content = raw.get("message", {}).get("content", "")
    finish_reason = raw.get("done_reason") or "stop"
    usage = _usage_from_ollama(raw)
    return _build_openai_chat_response(
        model=model,
        content=content,
        finish_reason=finish_reason,
        usage=usage,
    )


async def _stream_vllm(
    payload: dict[str, Any],
    settings: Settings,
    observation: dict[str, Any] | None,
) -> AsyncIterator[str]:
    timeout = httpx.Timeout(settings.request_timeout_seconds, read=None)
    collected_chunks: list[str] = []
    final_usage: dict[str, Any] | None = None

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                f"{settings.vllm_base_url}/v1/chat/completions",
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue

                    if line.startswith("data:"):
                        content = line[5:].strip()
                        if content == "[DONE]":
                            yield "data: [DONE]\n\n"
                            break

                        try:
                            chunk = json.loads(content)
                            text_piece = _extract_content_from_openai_chunk(chunk)
                            if text_piece:
                                collected_chunks.append(text_piece)
                            if isinstance(chunk.get("usage"), dict):
                                final_usage = chunk["usage"]
                        except json.JSONDecodeError:
                            pass

                        yield f"{line}\n\n"
                    else:
                        yield f"{line}\n\n"
    except Exception as exc:
        _finish_langfuse_observation(
            observation,
            output={"error": str(exc)},
            error=str(exc),
        )
        raise _http_exception_from_error(exc)

    _finish_langfuse_observation(
        observation,
        output={"content": "".join(collected_chunks)},
        usage=final_usage,
    )


async def _stream_ollama(
    payload: dict[str, Any],
    settings: Settings,
    model: str,
    observation: dict[str, Any] | None,
) -> AsyncIterator[str]:
    timeout = httpx.Timeout(settings.request_timeout_seconds, read=None)
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    generated_text: list[str] = []
    final_reason = "stop"
    usage: dict[str, int] | None = None

    try:
        ollama_payload = _build_ollama_payload(payload, model, stream=True)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                f"{settings.ollama_base_url}/api/chat",
                json=ollama_payload,
            ) as response:
                response.raise_for_status()

                role_chunk = _build_openai_chunk(
                    completion_id=completion_id,
                    model=model,
                    delta={"role": "assistant"},
                    finish_reason=None,
                )
                yield f"data: {json.dumps(role_chunk)}\n\n"

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    chunk = json.loads(line)
                    message = chunk.get("message", {})
                    content_piece = message.get("content", "")
                    if content_piece:
                        generated_text.append(content_piece)
                        openai_chunk = _build_openai_chunk(
                            completion_id=completion_id,
                            model=model,
                            delta={"content": content_piece},
                            finish_reason=None,
                        )
                        yield f"data: {json.dumps(openai_chunk)}\n\n"

                    if chunk.get("done"):
                        final_reason = chunk.get("done_reason") or "stop"
                        usage = _usage_from_ollama(chunk)
                        break

                done_chunk = _build_openai_chunk(
                    completion_id=completion_id,
                    model=model,
                    delta={},
                    finish_reason=final_reason,
                )
                yield f"data: {json.dumps(done_chunk)}\n\n"
                yield "data: [DONE]\n\n"
    except Exception as exc:
        _finish_langfuse_observation(
            observation,
            output={"error": str(exc)},
            error=str(exc),
        )
        raise _http_exception_from_error(exc)

    _finish_langfuse_observation(
        observation,
        output={"content": "".join(generated_text)},
        usage=usage,
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, str]:
    """Informacion del servicio."""
    return {
        "service": "LLM Enterprise Platform API",
        "version": "0.1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "docs": "/curso/api/docs",
    }


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(payload: ChatCompletionRequest):
    payload_dict = payload.model_dump(exclude_none=True)

    settings = get_settings()
    provider = _validate_provider(settings)
    _validate_chat_payload(payload_dict)

    stream = bool(payload_dict.get("stream", False))
    model = str(payload_dict.get("model") or _default_model_for_provider(settings, provider))
    normalized_payload = _normalize_openai_payload(payload_dict, model, stream)

    observation = _start_langfuse_observation(
        settings,
        provider=provider,
        model=model,
        payload=normalized_payload,
    )

    if stream:
        if provider == "vllm":
            generator = _stream_vllm(normalized_payload, settings, observation)
        else:
            generator = _stream_ollama(normalized_payload, settings, model, observation)

        return StreamingResponse(
            generator,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        if provider == "vllm":
            response_payload = await _call_vllm_non_stream(normalized_payload, settings)
        else:
            response_payload = await _call_ollama_non_stream(normalized_payload, settings, model)
    except Exception as exc:
        _finish_langfuse_observation(
            observation,
            output={"error": str(exc)},
            error=str(exc),
        )
        raise _http_exception_from_error(exc)

    usage = response_payload.get("usage") if isinstance(response_payload, dict) else None
    _finish_langfuse_observation(
        observation,
        output=response_payload,
        usage=usage if isinstance(usage, dict) else None,
    )

    return JSONResponse(response_payload)
