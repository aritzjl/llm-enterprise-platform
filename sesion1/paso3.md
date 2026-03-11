# Sesion 1 - Paso 3


## 1) Preparacion

Partimos desde la rama del paso 2 y creamos la rama del paso 3:

```bash
git switch sesion1/paso2
git switch -c sesion1/paso3
```

## 2) Cambios de codigo archivo por archivo

### 2.1 `api/app/schemas/chat.py`

#### Que cambia
- Se agregan dos campos obligatorios al schema de entrada: `provider` y `base_url`.
- `provider` se restringe a `"vllm"` o `"ollama"`.
- `max_tokens` pasa de obligatorio a opcional.
- `to_domain()` ahora propaga `provider` y `base_url` al modelo de dominio.

#### Por que
- Cada request debe decidir explicitamente contra que backend OpenAI-compatible se ejecuta.
- Necesitamos poder comparar vLLM y Ollama en la misma API sin cambiar variables globales.
- Hacer `max_tokens` opcional mejora compatibilidad con modelos que gestionan ese valor por defecto.

#### Contenido final completo

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
    """Subset of OpenAI chat completion payload supported in session 1."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "provider": "ollama",
                "base_url": "http://localhost:11436/v1",
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
    model: str = Field(
        min_length=1,
        description="Modelo obligatorio que se usara en esta llamada.",
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
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
```

### 2.2 `api/app/models/chat.py`

#### Que cambia
- `ChatRequest` incorpora `provider` y `base_url` como campos del dominio.
- `max_tokens` pasa a `int | None` con default `None`.

#### Por que
- El dominio debe transportar toda la informacion necesaria hasta el agente y la observabilidad.
- Si el schema acepta `max_tokens` opcional, el modelo de dominio debe reflejarlo.

#### Contenido final completo

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
    model: str
    messages: list[ChatMessage]
    max_tokens: int | None = None
    temperature: float | None = None
```

### 2.3 `api/app/api/v1/endpoints/llm.py`

#### Que cambia
- `GET /v1/models` pasa a recibir `base_url` obligatorio por query param.
- `GET /v1/models` acepta `provider` opcional por query param.
- El endpoint delega en `llm_service.list_models(base_url=..., provider=...)`.

#### Por que
- El listado de modelos tambien debe ser por request para comparar proveedores en caliente.
- `provider` opcional permite etiquetar la consulta y mantener consistencia semantica con chat.

#### Contenido final completo

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
    """Generate a chat completion with a LangGraph-backed single-node agent."""
    request = payload.to_domain()
    return llm_service.create_chat_completion(request)
```

### 2.4 `api/app/clients/openai_client.py`

#### Que cambia
- Se reemplaza el cliente singleton por una factoria cacheada por `base_url`.
- Se introduce `_build_openai_client(base_url, api_key, timeout)` con `@lru_cache(maxsize=16)`.
- `get_openai_client(base_url=None)` selecciona y normaliza la URL por request.

#### Por que
- Permite alternar vLLM/Ollama entre requests sin reiniciar la API.
- Mantiene beneficio de cache para no recrear cliente en cada llamada.

#### Contenido final completo

```python
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
```

### 2.5 `api/app/agents/chat_agent.py`

#### Que cambia
- El estado del agente incorpora `provider` y `base_url`.
- `max_tokens` en estado pasa a opcional.
- La llamada a OpenAI usa `get_openai_client(base_url=state["base_url"])`.
- Solo se envia `max_tokens` al payload cuando no es `None`.
- `create_llm_generation(...)` ahora recibe `provider` y `base_url`.

#### Por que
- El agente es el punto donde se ejecuta la inferencia real; debe usar el backend elegido en la request.
- Evitamos enviar parametros innecesarios al proveedor cuando no vienen informados.
- La generacion de Langfuse debe quedar etiquetada por proveedor/base URL.

#### Contenido final completo

```python
"""Single-node LangGraph chat agent."""

from time import perf_counter
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from app.clients.openai_client import get_openai_client
from app.models.chat import ChatRequest
from app.observability.langfuse import (
    create_llm_generation,
    end_llm_generation_error,
    end_llm_generation_success,
)


class ChatAgentState(TypedDict, total=False):
    """State passed through the LangGraph pipeline."""

    provider: str
    base_url: str
    model: str
    messages: list[dict[str, str]]
    temperature: float | None
    max_tokens: int | None
    completion: dict[str, Any]
    trace: Any | None


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


def _call_llm(state: ChatAgentState) -> ChatAgentState:
    start = perf_counter()
    generation = create_llm_generation(
        trace=state.get("trace"),
        provider=state["provider"],
        base_url=state["base_url"],
        model=state["model"],
        messages=state["messages"],
        temperature=state.get("temperature"),
        max_tokens=state["max_tokens"],
    )

    payload: dict[str, Any] = {
        "model": state["model"],
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
        )
        raise

    completion_json = completion.model_dump(mode="json")
    end_llm_generation_success(
        generation=generation,
        output=completion_json,
        usage_details=_extract_usage_details(completion_json),
        latency_ms=int((perf_counter() - start) * 1000),
    )
    return {"completion": completion_json}


def _build_chat_agent():
    builder = StateGraph(ChatAgentState)
    builder.add_node("call_llm", _call_llm)
    builder.set_entry_point("call_llm")
    builder.add_edge("call_llm", END)
    return builder.compile()


_chat_agent = None


def initialize_chat_agent() -> None:
    """Compile the graph once and cache it for the app lifetime."""
    global _chat_agent
    if _chat_agent is None:
        _chat_agent = _build_chat_agent()


def run_chat_agent(request: ChatRequest, trace: Any | None = None) -> dict[str, Any]:
    """Invoke the precompiled graph for one chat completion request."""
    global _chat_agent
    if _chat_agent is None:
        initialize_chat_agent()

    state: ChatAgentState = {
        "provider": request.provider,
        "base_url": request.base_url,
        "model": request.model,
        "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "trace": trace,
    }
    result = _chat_agent.invoke(state)
    return result["completion"]
```

### 2.6 `api/app/observability/langfuse.py`

#### Que cambia
- `create_chat_trace()` agrega `provider` y `base_url` en `trace.input` y `trace.metadata`.
- `update_chat_trace_success()` y `update_chat_trace_error()` reciben y guardan `provider/base_url`.
- `create_llm_generation()` recibe `provider/base_url` y los guarda en metadata.
- `model_parameters` solo incluye `max_tokens` si existe.

#### Por que
- La comparativa vLLM vs Ollama depende de poder filtrar trazas por proveedor y URL real usada.
- Guardar el mismo contexto en exito y error evita trazas incompletas al depurar incidencias.

#### Contenido final completo

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
) -> None:
    """Attach successful output metadata to trace."""
    if trace is None:
        return

    try:
        trace.update(
            output=completion,
            metadata={
                "status": "success",
                "latency_ms": latency_ms,
                "provider": provider,
                "base_url": base_url,
            },
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Failed to update Langfuse trace (success): %s", exc)


def update_chat_trace_error(
    trace: Any | None,
    error_message: str,
    latency_ms: int,
    provider: str,
    base_url: str,
) -> None:
    """Attach error metadata to trace."""
    if trace is None:
        return

    try:
        trace.update(
            metadata={
                "status": "error",
                "latency_ms": latency_ms,
                "provider": provider,
                "base_url": base_url,
            },
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
        return trace.generation(
            name="llm.generation",
            model=model,
            input=messages,
            model_parameters=model_parameters,
            metadata={"provider": provider, "base_url": base_url},
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
) -> None:
    """Mark generation as successful."""
    if generation is None:
        return

    try:
        generation.end(
            output=output,
            usage_details=usage_details,
            metadata={"status": "success", "latency_ms": latency_ms},
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Failed to end Langfuse generation (success): %s", exc)


def end_llm_generation_error(
    generation: Any | None, *, error_message: str, latency_ms: int
) -> None:
    """Mark generation as failed."""
    if generation is None:
        return

    try:
        generation.end(
            level="ERROR",
            status_message=error_message,
            metadata={"status": "error", "latency_ms": latency_ms},
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Failed to end Langfuse generation (error): %s", exc)
```

### 2.7 `api/app/services/llm_service.py`

#### Que cambia
- `list_models()` ahora recibe `base_url` y `provider`.
- El servicio crea cliente OpenAI con `base_url` dinamica por request.
- Los errores de `list_models` incluyen `provider/base_url` en el mensaje.
- Al actualizar traza de chat (exito/error) se propagan `provider` y `base_url`.

#### Por que
- El servicio es la capa que orquesta cliente y errores upstream; debe tener contexto completo de proveedor.
- Trazas y errores con contexto facilitan diagnostico en ejercicios de comparativa.

#### Contenido final completo

```python
"""Service layer for model listing and chat completion generation."""

from time import perf_counter
from typing import Any

from openai import APIConnectionError, APIError, APITimeoutError

from app.agents.chat_agent import run_chat_agent
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
    try:
        completion = run_chat_agent(request, trace=trace)
    except (APIConnectionError, APITimeoutError, APIError) as exc:
        update_chat_trace_error(
            trace=trace,
            error_message=str(exc),
            latency_ms=int((perf_counter() - start) * 1000),
            provider=request.provider,
            base_url=request.base_url,
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
        )
        raise

    update_chat_trace_success(
        trace=trace,
        completion=completion,
        latency_ms=int((perf_counter() - start) * 1000),
        provider=request.provider,
        base_url=request.base_url,
    )
    return completion
```


Durante el curso transformaremos esta base en:

- Una plataforma LLM corporativa completa
- RAG agentico avanzado
- Evaluacion automatica con LLM-as-judge
- Multi-tenant
- Integracion con CRM
- Optimizacion y escalabilidad
- Arquitectura enterprise final
```

## 3) Build y arranque de API

```bash
docker compose build api
docker compose up -d api
```

## 4) Pruebas funcionales con curl

### 4.1 Listado de modelos por proveedor

```bash
curl 'http://localhost:8000/curso/api/v1/models?provider=vllm&base_url=http://localhost:8001/v1'
curl 'http://localhost:8000/curso/api/v1/models?provider=ollama&base_url=http://localhost:11436/v1'
```

### 4.2 Chat con vLLM

```bash
curl -X POST 'http://localhost:8000/curso/api/v1/chat/completions' \
  -H 'Content-Type: application/json' \
  -d '{
    "provider": "vllm",
    "base_url": "http://localhost:8001/v1",
    "model": "Qwen/Qwen2-1.5B-Instruct-AWQ",
    "messages": [{"role":"user","content":"Responde con VLLM"}],
    "max_tokens": 32,
    "temperature": 0.2,
    "stream": false
  }'
```

### 4.3 Chat con Ollama

```bash
curl -X POST 'http://localhost:8000/curso/api/v1/chat/completions' \
  -H 'Content-Type: application/json' \
  -d '{
    "provider": "ollama",
    "base_url": "http://localhost:11436/v1",
    "model": "llama3.1:8b",
    "messages": [{"role":"user","content":"Responde con OLLAMA"}],
    "max_tokens": 32,
    "temperature": 0.2,
    "stream": false
  }'
```

### 4.4 Chat sin `max_tokens` (validar que ahora es opcional)

```bash
curl -X POST 'http://localhost:8000/curso/api/v1/chat/completions' \
  -H 'Content-Type: application/json' \
  -d '{
    "provider": "ollama",
    "base_url": "http://localhost:11436/v1",
    "model": "llama3.1:8b",
    "messages": [{"role":"user","content":"Responde en una frase"}],
    "temperature": 0.2,
    "stream": false
  }'
```

## 5) Validacion en Langfuse

En `https://cloud.langfuse.com`:
- Filtra por `environment=sesion1-paso2` (o el que uses en tu `.env`).
- Abre trazas de `chat.completions`.
- Verifica que en trace y generation aparezcan:
  - `provider`
  - `base_url`
- Comprueba que tambien quedan registrados en llamadas con error (si fuerzas una URL invalida).

## 6) Commit final

```bash
git add sesion1/paso3.md
git commit -m "docs: guia paso a paso para pasar de sesion1/paso2 a sesion1/paso3"
```
