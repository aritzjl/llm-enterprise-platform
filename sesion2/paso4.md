# Sesión 2 - Paso 4

Guía práctica para ampliar el proyecto con:
- Modelo enrutador (`router_model`) configurable por request que clasifica peticiones como `simple` o `complex`
- Agente LangGraph modular refactorizado a paquete profesional (`state`, `prompts`, `nodes`, `graph`)
- Ruta simple → responde el modelo pequeño (router); ruta compleja → responde el modelo grande
- Observabilidad extendida en Langfuse con trazas por generación (`llm.router.classification`, `llm.router.response`, `llm.generation`)

## 1) Prerrequisitos

- Tener completado **sesion1/paso3** (o partir de la rama `sesion1/paso3`):

```bash
git switch sesion1/paso3
git switch -c sesion2/paso4
```

- Tener Docker/Compose operativo:

```bash
docker --version
docker compose version
```

## 2) Crear estructura del nuevo paquete de agente

Desde la raíz del repo:

```bash
mkdir -p api/app/agents/chat_completion_agent
touch api/app/agents/chat_completion_agent/__init__.py
touch api/app/agents/chat_completion_agent/state.py
touch api/app/agents/chat_completion_agent/prompts.py
touch api/app/agents/chat_completion_agent/nodes.py
touch api/app/agents/chat_completion_agent/graph.py
```

## 3) Eliminar el agente monolítico antiguo

```bash
rm api/app/agents/chat_agent.py
```

## 4) Reemplazar archivos (archivo por archivo)

Copia-pega el contenido completo de cada archivo. Si el archivo ya existe, reemplázalo entero.

## 4.1 `api/app/models/chat.py`

```python
"""Domain models for chat use cases."""

from dataclasses import dataclass


@dataclass(slots=True)
class ChatMessage:
    """Single chat message in domain format."""

    role: str
    content: str


@dataclass(slots=True)
class ChatRequest:
    """Domain request for a single non-stream chat completion."""

    provider: str
    base_url: str
    router_model: str
    model: str
    messages: list[ChatMessage]
    max_tokens: int | None = None
    temperature: float | None = None
```

## 4.2 `api/app/schemas/chat.py`

```python
"""HTTP schemas for OpenAI-compatible chat completion."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models.chat import ChatMessage, ChatRequest


class ChatMessageSchema(BaseModel):
    """Input message shape for chat completion requests."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ChatCompletionRequestSchema(BaseModel):
    """Subset of OpenAI chat completion payload supported in this project."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "provider": "ollama",
                "base_url": "http://localhost:11436/v1",
                "router_model": "llama3.2:1b",
                "model": "llama3.1:8b",
                "messages": [
                    {"role": "user", "content": "Explica LangGraph en una frase corta."}
                ],
                "temperature": 0.2,
                "stream": False,
            }
        },
    )

    provider: Literal["vllm", "ollama"] = Field(
        description="Proveedor de inferencia para esta llamada (vllm u ollama).",
    )
    base_url: str = Field(
        min_length=1,
        description="Base URL OpenAI-compatible del proveedor para esta llamada.",
        examples=["http://localhost:8001/v1", "http://localhost:11436/v1"],
    )
    router_model: str = Field(
        min_length=1,
        description="Modelo enrutador obligatorio para clasificar simple/complex.",
        examples=["llama3.2:1b"],
    )
    model: str = Field(
        min_length=1,
        description="Modelo principal obligatorio para consultas complejas.",
        examples=["llama3.1:8b"],
    )
    messages: list[ChatMessageSchema] = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1)
    stream: Literal[False] = False

    def to_domain(self) -> ChatRequest:
        """Convert transport schema into domain request model."""
        messages = [ChatMessage(role=m.role, content=m.content) for m in self.messages]
        return ChatRequest(
            provider=self.provider,
            base_url=self.base_url.strip(),
            router_model=self.router_model.strip(),
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
```

## 4.3 `api/app/agents/chat_completion_agent/state.py`

```python
"""State contracts for the chat completion LangGraph agent."""

from typing import Any, Literal, TypedDict

ChatAgentRoute = Literal["simple", "complex"]


class ChatAgentState(TypedDict, total=False):
    """State passed through the LangGraph pipeline."""

    provider: str
    base_url: str
    router_model: str
    model: str
    messages: list[dict[str, str]]
    temperature: float | None
    max_tokens: int | None
    selected_route: ChatAgentRoute
    responder_model: str
    completion: dict[str, Any]
    trace: Any | None


class ChatAgentResult(TypedDict):
    """Public result returned by the chat completion agent."""

    completion: dict[str, Any]
    selected_route: ChatAgentRoute
    responder_model: str
```

## 4.4 `api/app/agents/chat_completion_agent/prompts.py`

```python
"""Prompt builders for the chat completion LangGraph agent."""

import json

ROUTER_SYSTEM_PROMPT = (
    "You are a routing model. Classify whether the user request is SIMPLE or COMPLEX. "
    "SIMPLE: short factual requests, greetings, quick formatting, single-step asks. "
    "COMPLEX: multi-step reasoning, planning, deep comparisons, coding/debugging, "
    "or requests requiring detailed structured output. "
    "Return ONLY strict JSON in one line with this exact schema: "
    '{"route":"simple"} or {"route":"complex"}. '
    "No extra keys, no markdown, no prose."
)


def build_router_classification_messages(
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Build messages used by the router model for simple/complex classification."""
    conversation_json = json.dumps(messages, ensure_ascii=True)
    user_prompt = (
        "Classify this conversation as simple or complex. "
        f"Conversation JSON: {conversation_json}"
    )
    return [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
```

## 4.5 `api/app/agents/chat_completion_agent/nodes.py`

```python
"""LangGraph nodes for chat completion with router-first execution."""

from __future__ import annotations

import json
import re
from time import perf_counter
from typing import Any

from app.clients.openai_client import get_openai_client
from app.observability.langfuse import (
    create_llm_generation,
    end_llm_generation_error,
    end_llm_generation_success,
)

from .prompts import build_router_classification_messages
from .state import ChatAgentRoute, ChatAgentState

ROUTER_CLASSIFICATION_TEMPERATURE = 0.0
ROUTER_CLASSIFICATION_MAX_TOKENS = 32


def _extract_usage_details(completion: dict[str, Any]) -> dict[str, int] | None:
    usage = completion.get("usage")
    if not isinstance(usage, dict):
        return None

    details = {
        key: value
        for key, value in usage.items()
        if key in {"prompt_tokens", "completion_tokens", "total_tokens"}
        and isinstance(value, int)
    }
    return details or None


def _completion_content(completion: dict[str, Any]) -> str:
    choices = completion.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return ""
    message = first_choice.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _parse_route_from_content(content: str) -> ChatAgentRoute:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    candidate_payloads = [cleaned]
    json_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if json_match is not None:
        candidate_payloads.append(json_match.group(0))

    for payload in candidate_payloads:
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, dict):
            route = parsed.get("route")
            if route == "simple":
                return "simple"
            if route == "complex":
                return "complex"

    return "complex"


def route_selector(state: ChatAgentState) -> ChatAgentRoute:
    """Select branch after classification, defaulting to complex for safety."""
    return "simple" if state.get("selected_route") == "simple" else "complex"


def classify_route_node(state: ChatAgentState) -> ChatAgentState:
    """Classify conversation complexity with the router model."""
    start = perf_counter()
    router_messages = build_router_classification_messages(state["messages"])
    generation = create_llm_generation(
        trace=state.get("trace"),
        provider=state["provider"],
        base_url=state["base_url"],
        model=state["router_model"],
        messages=router_messages,
        temperature=ROUTER_CLASSIFICATION_TEMPERATURE,
        max_tokens=ROUTER_CLASSIFICATION_MAX_TOKENS,
        generation_name="llm.router.classification",
        metadata={
            "stage": "classification",
            "router_model": state["router_model"],
            "primary_model": state["model"],
        },
    )

    payload: dict[str, Any] = {
        "model": state["router_model"],
        "messages": router_messages,
        "stream": False,
        "temperature": ROUTER_CLASSIFICATION_TEMPERATURE,
        "max_tokens": ROUTER_CLASSIFICATION_MAX_TOKENS,
    }

    try:
        completion = get_openai_client(base_url=state["base_url"]).chat.completions.create(
            **payload
        )
    except Exception as exc:
        end_llm_generation_error(
            generation=generation,
            error_message=str(exc),
            latency_ms=int((perf_counter() - start) * 1000),
            extra_metadata={"stage": "classification"},
        )
        raise

    completion_json = completion.model_dump(mode="json")
    route = _parse_route_from_content(_completion_content(completion_json))

    end_llm_generation_success(
        generation=generation,
        output=completion_json,
        usage_details=_extract_usage_details(completion_json),
        latency_ms=int((perf_counter() - start) * 1000),
        extra_metadata={
            "stage": "classification",
            "selected_route": route,
        },
    )
    return {"selected_route": route}


def _call_response_model(
    state: ChatAgentState,
    *,
    target_model: str,
    generation_name: str,
    stage: str,
) -> ChatAgentState:
    start = perf_counter()
    generation = create_llm_generation(
        trace=state.get("trace"),
        provider=state["provider"],
        base_url=state["base_url"],
        model=target_model,
        messages=state["messages"],
        temperature=state.get("temperature"),
        max_tokens=state.get("max_tokens"),
        generation_name=generation_name,
        metadata={
            "stage": stage,
            "selected_route": state.get("selected_route", "complex"),
            "responder_model": target_model,
            "router_model": state["router_model"],
            "primary_model": state["model"],
        },
    )

    payload: dict[str, Any] = {
        "model": target_model,
        "messages": state["messages"],
        "stream": False,
    }
    if state.get("max_tokens") is not None:
        payload["max_tokens"] = state["max_tokens"]
    if state.get("temperature") is not None:
        payload["temperature"] = state["temperature"]

    try:
        completion = get_openai_client(base_url=state["base_url"]).chat.completions.create(
            **payload
        )
    except Exception as exc:
        end_llm_generation_error(
            generation=generation,
            error_message=str(exc),
            latency_ms=int((perf_counter() - start) * 1000),
            extra_metadata={
                "stage": stage,
                "selected_route": state.get("selected_route", "complex"),
                "responder_model": target_model,
            },
        )
        raise

    completion_json = completion.model_dump(mode="json")
    end_llm_generation_success(
        generation=generation,
        output=completion_json,
        usage_details=_extract_usage_details(completion_json),
        latency_ms=int((perf_counter() - start) * 1000),
        extra_metadata={
            "stage": stage,
            "selected_route": state.get("selected_route", "complex"),
            "responder_model": target_model,
        },
    )
    return {
        "completion": completion_json,
        "responder_model": target_model,
    }


def call_router_model_node(state: ChatAgentState) -> ChatAgentState:
    """Generate final response with router model for simple requests."""
    return _call_response_model(
        state,
        target_model=state["router_model"],
        generation_name="llm.router.response",
        stage="router_response",
    )


def call_primary_model_node(state: ChatAgentState) -> ChatAgentState:
    """Generate final response with primary model for complex requests."""
    return _call_response_model(
        state,
        target_model=state["model"],
        generation_name="llm.generation",
        stage="primary_response",
    )
```

## 4.6 `api/app/agents/chat_completion_agent/graph.py`

```python
"""Graph assembly and runtime for the chat completion agent."""

from __future__ import annotations

from typing import Any, cast

from langgraph.graph import END, StateGraph

from app.models.chat import ChatRequest

from .nodes import (
    call_primary_model_node,
    call_router_model_node,
    classify_route_node,
    route_selector,
)
from .state import ChatAgentResult, ChatAgentState


def _build_chat_agent():
    builder = StateGraph(ChatAgentState)
    builder.add_node("classify_route", classify_route_node)
    builder.add_node("call_router_model", call_router_model_node)
    builder.add_node("call_primary_model", call_primary_model_node)

    builder.set_entry_point("classify_route")
    builder.add_conditional_edges(
        "classify_route",
        route_selector,
        {
            "simple": "call_router_model",
            "complex": "call_primary_model",
        },
    )
    builder.add_edge("call_router_model", END)
    builder.add_edge("call_primary_model", END)
    return builder.compile()


_chat_agent = None


def initialize_chat_agent() -> None:
    """Compile the graph once and cache it for the app lifetime."""
    global _chat_agent
    if _chat_agent is None:
        _chat_agent = _build_chat_agent()


def run_chat_agent(request: ChatRequest, trace: Any | None = None) -> ChatAgentResult:
    """Invoke the precompiled graph for one chat completion request."""
    global _chat_agent
    if _chat_agent is None:
        initialize_chat_agent()

    state: ChatAgentState = {
        "provider": request.provider,
        "base_url": request.base_url,
        "router_model": request.router_model,
        "model": request.model,
        "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "trace": trace,
    }
    result = cast(ChatAgentState, _chat_agent.invoke(state))
    selected_route = route_selector(result)
    responder_model = result.get(
        "responder_model",
        request.router_model if selected_route == "simple" else request.model,
    )

    return {
        "completion": result["completion"],
        "selected_route": selected_route,
        "responder_model": responder_model,
    }
```

## 4.7 `api/app/agents/chat_completion_agent/__init__.py`

```python
"""Chat completion agent package."""

from .graph import initialize_chat_agent, run_chat_agent

__all__ = ["initialize_chat_agent", "run_chat_agent"]
```

## 4.8 `api/app/agents/__init__.py`

```python
"""LangGraph agent modules."""

from app.agents.chat_completion_agent import initialize_chat_agent, run_chat_agent

__all__ = ["initialize_chat_agent", "run_chat_agent"]
```

## 4.9 `api/app/observability/langfuse.py`

```python
"""Langfuse helpers for best-effort observability."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from langfuse import Langfuse

from app.core.settings import get_settings
from app.models.chat import ChatRequest

logger = logging.getLogger(__name__)


def _langfuse_enabled() -> bool:
    settings = get_settings()
    return bool(settings.langfuse_public_key and settings.langfuse_secret_key)


@lru_cache(maxsize=1)
def get_langfuse_client() -> Langfuse | None:
    """Return a cached Langfuse client or None when observability is disabled."""
    settings = get_settings()
    if not _langfuse_enabled():
        logger.warning(
            "Langfuse disabled: LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY are missing."
        )
        return None

    try:
        return Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_base_url,
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Failed to initialize Langfuse client: %s", exc)
        return None


def flush_langfuse() -> None:
    """Flush pending telemetry without failing API shutdown."""
    client = get_langfuse_client()
    if client is None:
        return

    try:
        client.flush()
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Failed to flush Langfuse events: %s", exc)


def create_chat_trace(request: ChatRequest) -> Any | None:
    """Create a trace for one chat completion request."""
    client = get_langfuse_client()
    if client is None:
        return None

    settings = get_settings()
    trace_input = {
        "provider": request.provider,
        "base_url": request.base_url,
        "router_model": request.router_model,
        "model": request.model,
        "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
    }

    try:
        return client.trace(
            name="chat.completions",
            input=trace_input,
            metadata={
                "endpoint": "/v1/chat/completions",
                "provider": request.provider,
                "base_url": request.base_url,
                "router_model": request.router_model,
            },
            environment=settings.langfuse_environment,
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Failed to create Langfuse trace: %s", exc)
        return None


def update_chat_trace_success(
    trace: Any | None,
    completion: dict[str, Any],
    latency_ms: int,
    provider: str,
    base_url: str,
    router_model: str,
    selected_route: str | None = None,
    responder_model: str | None = None,
) -> None:
    """Attach successful output metadata to trace."""
    if trace is None:
        return

    try:
        metadata: dict[str, Any] = {
            "status": "success",
            "latency_ms": latency_ms,
            "provider": provider,
            "base_url": base_url,
            "router_model": router_model,
        }
        if selected_route is not None:
            metadata["selected_route"] = selected_route
        if responder_model is not None:
            metadata["responder_model"] = responder_model

        trace.update(
            output=completion,
            metadata=metadata,
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Failed to update Langfuse trace (success): %s", exc)


def update_chat_trace_error(
    trace: Any | None,
    error_message: str,
    latency_ms: int,
    provider: str,
    base_url: str,
    router_model: str,
    selected_route: str | None = None,
    responder_model: str | None = None,
) -> None:
    """Attach error metadata to trace."""
    if trace is None:
        return

    try:
        metadata: dict[str, Any] = {
            "status": "error",
            "latency_ms": latency_ms,
            "provider": provider,
            "base_url": base_url,
            "router_model": router_model,
        }
        if selected_route is not None:
            metadata["selected_route"] = selected_route
        if responder_model is not None:
            metadata["responder_model"] = responder_model

        trace.update(
            metadata=metadata,
            output={"error": error_message},
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Failed to update Langfuse trace (error): %s", exc)


def create_llm_generation(
    trace: Any | None,
    *,
    provider: str,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float | None,
    max_tokens: int | None,
    generation_name: str = "llm.generation",
    metadata: dict[str, Any] | None = None,
) -> Any | None:
    """Create a child generation for one upstream LLM call."""
    if trace is None:
        return None

    settings = get_settings()
    model_parameters: dict[str, Any] = {}
    if max_tokens is not None:
        model_parameters["max_tokens"] = max_tokens
    if temperature is not None:
        model_parameters["temperature"] = temperature

    try:
        generation_metadata = {"provider": provider, "base_url": base_url}
        if metadata is not None:
            generation_metadata.update(metadata)

        return trace.generation(
            name=generation_name,
            model=model,
            input=messages,
            model_parameters=model_parameters,
            metadata=generation_metadata,
            environment=settings.langfuse_environment,
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Failed to create Langfuse generation: %s", exc)
        return None


def end_llm_generation_success(
    generation: Any | None,
    *,
    output: dict[str, Any],
    usage_details: dict[str, int] | None,
    latency_ms: int,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    """Mark generation as successful."""
    if generation is None:
        return

    try:
        metadata: dict[str, Any] = {"status": "success", "latency_ms": latency_ms}
        if extra_metadata is not None:
            metadata.update(extra_metadata)

        generation.end(
            output=output,
            usage_details=usage_details,
            metadata=metadata,
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Failed to end Langfuse generation (success): %s", exc)


def end_llm_generation_error(
    generation: Any | None,
    *,
    error_message: str,
    latency_ms: int,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    """Mark generation as failed."""
    if generation is None:
        return

    try:
        metadata: dict[str, Any] = {"status": "error", "latency_ms": latency_ms}
        if extra_metadata is not None:
            metadata.update(extra_metadata)

        generation.end(
            level="ERROR",
            status_message=error_message,
            metadata=metadata,
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Failed to end Langfuse generation (error): %s", exc)
```

## 4.10 `api/app/services/llm_service.py`

```python
"""Service layer for model listing and chat completion generation."""

from time import perf_counter
from typing import Any

from openai import APIConnectionError, APIError, APITimeoutError

from app.agents.chat_completion_agent import run_chat_agent
from app.clients.openai_client import get_openai_client
from app.core.exceptions import UpstreamServiceError
from app.models.chat import ChatRequest
from app.observability.langfuse import (
    create_chat_trace,
    update_chat_trace_error,
    update_chat_trace_success,
)


def list_models(base_url: str, provider: str | None = None) -> dict[str, Any]:
    """Proxy model listing from a request-selected OpenAI-compatible provider."""
    client = get_openai_client(base_url=base_url)
    try:
        response = client.models.list()
    except (APIConnectionError, APITimeoutError, APIError) as exc:
        provider_label = provider or "unknown"
        raise UpstreamServiceError(
            f"Error listing models from upstream provider ({provider_label}, {base_url}): {exc}"
        ) from exc
    return response.model_dump(mode="json")


def create_chat_completion(request: ChatRequest) -> dict[str, Any]:
    """Generate one chat completion via LangGraph."""
    trace = create_chat_trace(request)
    start = perf_counter()
    selected_route: str | None = None
    responder_model: str | None = None
    try:
        agent_result = run_chat_agent(request, trace=trace)
        completion = agent_result["completion"]
        selected_route = agent_result["selected_route"]
        responder_model = agent_result["responder_model"]
    except (APIConnectionError, APITimeoutError, APIError) as exc:
        update_chat_trace_error(
            trace=trace,
            error_message=str(exc),
            latency_ms=int((perf_counter() - start) * 1000),
            provider=request.provider,
            base_url=request.base_url,
            router_model=request.router_model,
            selected_route=selected_route,
            responder_model=responder_model,
        )
        raise UpstreamServiceError(
            f"Error generating chat completion from upstream provider: {exc}"
        ) from exc
    except Exception as exc:
        update_chat_trace_error(
            trace=trace,
            error_message=str(exc),
            latency_ms=int((perf_counter() - start) * 1000),
            provider=request.provider,
            base_url=request.base_url,
            router_model=request.router_model,
            selected_route=selected_route,
            responder_model=responder_model,
        )
        raise

    update_chat_trace_success(
        trace=trace,
        completion=completion,
        latency_ms=int((perf_counter() - start) * 1000),
        provider=request.provider,
        base_url=request.base_url,
        router_model=request.router_model,
        selected_route=selected_route,
        responder_model=responder_model,
    )
    return completion
```

## 4.11 `api/app/main.py`

```python
"""FastAPI bootstrap for the LLM Enterprise Platform API."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.agents.chat_completion_agent import initialize_chat_agent
from app.api.v1.router import router as api_v1_router
from app.core.exceptions import UpstreamServiceError
from app.core.settings import get_settings
from app.observability.langfuse import flush_langfuse

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    root_path=settings.app_root_path,
)
app.include_router(api_v1_router)


@app.on_event("startup")
def on_startup() -> None:
    """Initialize long-lived app components."""
    initialize_chat_agent()


@app.on_event("shutdown")
def on_shutdown() -> None:
    """Flush observability events before process exit."""
    flush_langfuse()


@app.exception_handler(UpstreamServiceError)
def handle_upstream_service_error(
    _request: Request, exc: UpstreamServiceError
) -> JSONResponse:
    """Map upstream provider errors to a consistent API response."""
    return JSONResponse(status_code=502, content={"detail": exc.detail})
```

## 4.12 `api/app/api/v1/endpoints/llm.py`

```python
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
```

## 5) Ejecutar la API

### Opción A: Local

```bash
cd api
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Opción B: Docker

```bash
docker compose build api
docker compose --profile ollama up -d
docker compose logs -f api
```

## 6) Probar endpoints

## 6.1 Health

```bash
curl http://localhost:8000/curso/api/health
```

Esperado:

```json
{"status":"ok"}
```

## 6.2 Listar modelos

```bash
curl 'http://localhost:8000/curso/api/v1/models?provider=ollama&base_url=http://localhost:11436/v1'
```

Esperado: JSON tipo OpenAI con campo `data`.

## 6.3 Chat completion - ruta simple (responde el router_model)

```bash
curl -X POST 'http://localhost:8000/curso/api/v1/chat/completions' \
  -H 'Content-Type: application/json' \
  -d '{
    "provider": "ollama",
    "base_url": "http://localhost:11436/v1",
    "router_model": "llama3.2:1b",
    "model": "llama3.1:8b",
    "messages": [{"role":"user","content":"Dime hola en una frase"}],
    "max_tokens": 32,
    "temperature": 0.2,
    "stream": false
  }'
```

Esperado en Langfuse: `selected_route = simple`, `responder_model = llama3.2:1b`.

## 6.4 Chat completion - ruta compleja (responde el modelo grande)

```bash
curl -X POST 'http://localhost:8000/curso/api/v1/chat/completions' \
  -H 'Content-Type: application/json' \
  -d '{
    "provider": "ollama",
    "base_url": "http://localhost:11436/v1",
    "router_model": "llama3.2:1b",
    "model": "llama3.1:8b",
    "messages": [{"role":"user","content":"Compara LangGraph y LangChain con criterios tecnicos y dame una recomendacion en tabla"}],
    "max_tokens": 256,
    "temperature": 0.2,
    "stream": false
  }'
```

Esperado en Langfuse: `selected_route = complex`, `responder_model = llama3.1:8b`.

## 6.5 Chat completion inválido (falta `router_model`)

```bash
curl -X POST 'http://localhost:8000/curso/api/v1/chat/completions' \
  -H 'Content-Type: application/json' \
  -d '{
    "provider": "ollama",
    "base_url": "http://localhost:11436/v1",
    "model": "llama3.1:8b",
    "messages": [{"role":"user","content":"hola"}],
    "stream": false
  }'
```

Esperado: `422` por campo obligatorio `router_model`.

## 7) Validación en Langfuse

En `https://cloud.langfuse.com`:
- Filtrar por el `environment` configurado en `.env`.
- Validar en trace metadata/input:
  - `provider`
  - `base_url`
  - `router_model`
  - `selected_route`
  - `responder_model`
- Validar generaciones:
  - `llm.router.classification` (siempre presente)
  - `llm.router.response` (si ruta simple)
  - `llm.generation` (si ruta complex)

## 8) Troubleshooting rápido

- Si la clasificación siempre devuelve `complex`:
  - El modelo router puede no ser capaz de generar JSON estricto. Prueba con un modelo más capaz como `router_model`.
  - El fallback por diseño es `complex` ante cualquier respuesta no parseable.

- Si `/v1/models` falla:
  - Verifica que Ollama está arriba:
    ```bash
    curl http://localhost:11436/v1/models
    ```
  - Revisa que pasas `base_url` como query param.

- Si cambiaste dependencias y no se reflejan en Docker:
  ```bash
  docker compose build --no-cache api
  docker compose --profile ollama up -d
  ```

- Si no arranca FastAPI por imports:
  - Asegura que ejecutas desde `api/` cuando lanzas `uvicorn app.main:app`.
  - Verifica que eliminaste `api/app/agents/chat_agent.py` y que el nuevo paquete `chat_completion_agent/` tiene su `__init__.py`.
