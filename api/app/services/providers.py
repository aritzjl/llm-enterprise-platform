from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator

import httpx
from fastapi import HTTPException

from app.core.config import Settings
from app.services.observability import finish_observation


def validate_chat_payload(payload: dict[str, Any]) -> None:
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="Field 'messages' must be a non-empty list.")


def normalize_openai_payload(payload: dict[str, Any], model: str, stream: bool) -> dict[str, Any]:
    request_payload = dict(payload)
    request_payload["model"] = model
    request_payload["stream"] = stream
    return request_payload


def build_ollama_payload(payload: dict[str, Any], model: str, stream: bool) -> dict[str, Any]:
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


def http_exception_from_error(exc: Exception) -> HTTPException:
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


async def call_vllm_non_stream(payload: dict[str, Any], settings: Settings) -> dict[str, Any]:
    timeout = httpx.Timeout(settings.request_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(f"{settings.vllm_base_url}/v1/chat/completions", json=payload)
        response.raise_for_status()
        return response.json()


async def call_ollama_non_stream(
    payload: dict[str, Any],
    settings: Settings,
    model: str,
) -> dict[str, Any]:
    timeout = httpx.Timeout(settings.request_timeout_seconds)
    ollama_payload = build_ollama_payload(payload, model, stream=False)

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


async def stream_vllm(
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
        finish_observation(
            observation,
            output={"error": str(exc)},
            error=str(exc),
        )
        raise http_exception_from_error(exc)

    finish_observation(
        observation,
        output={"content": "".join(collected_chunks)},
        usage=final_usage,
    )


async def stream_ollama(
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
        ollama_payload = build_ollama_payload(payload, model, stream=True)
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
        finish_observation(
            observation,
            output={"error": str(exc)},
            error=str(exc),
        )
        raise http_exception_from_error(exc)

    finish_observation(
        observation,
        output={"content": "".join(generated_text)},
        usage=usage,
    )
