from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from app.core.config import (
    default_model_for_provider,
    get_settings,
    validate_provider,
)
from app.schemas.chat import ChatCompletionRequest
from app.services import providers
from app.services.observability import finish_observation, start_observation

router = APIRouter(tags=["chat"])


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(payload: ChatCompletionRequest):
    payload_dict = payload.model_dump(exclude_none=True)

    settings = get_settings()
    provider = validate_provider(settings)
    providers.validate_chat_payload(payload_dict)

    stream = bool(payload_dict.get("stream", False))
    model = str(payload_dict.get("model") or default_model_for_provider(settings, provider))
    normalized_payload = providers.normalize_openai_payload(payload_dict, model, stream)

    observation = start_observation(
        settings,
        provider=provider,
        model=model,
        payload=normalized_payload,
    )

    if stream:
        if provider == "vllm":
            generator = providers.stream_vllm(normalized_payload, settings, observation)
        else:
            generator = providers.stream_ollama(normalized_payload, settings, model, observation)

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
            response_payload = await providers.call_vllm_non_stream(normalized_payload, settings)
        else:
            response_payload = await providers.call_ollama_non_stream(normalized_payload, settings, model)
    except Exception as exc:
        finish_observation(
            observation,
            output={"error": str(exc)},
            error=str(exc),
        )
        raise providers.http_exception_from_error(exc)

    usage = response_payload.get("usage") if isinstance(response_payload, dict) else None
    finish_observation(
        observation,
        output=response_payload,
        usage=usage if isinstance(usage, dict) else None,
    )

    return JSONResponse(response_payload)
