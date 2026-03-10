# Sesión 1 - Paso 1

Guía práctica para ampliar el proyecto con:
- FastAPI en estructura estándar por capas (`api`, `core`, `schemas`, `services`, `models`, `clients`, `agents`)
- Agente LangGraph (1 nodo)
- Cliente OpenAI con `base_url` configurable desde `.env`
- Endpoints OpenAI-like:
  - `GET /v1/models`
  - `POST /v1/chat/completions`

## 1) Prerrequisitos

- Tener el repo clonado y entrar en la raíz del proyecto:

```bash
cd llm-enterprise-platform
```

- Tener Docker/Compose operativo (si trabajas con contenedores):

```bash
docker --version
docker compose version
```

- Opcional para ejecutar API en local:
  - Python 3.11+

## 2) Configurar variables de entorno

Si aún no existe `.env`:

```bash
cp .env.example .env
```

Añade (o verifica) estas variables en `.env`:

```dotenv
OPENAI_BASE_URL=http://localhost:8001/v1
OPENAI_API_KEY=dummy
OPENAI_TIMEOUT_SECONDS=60
```

## 3) Instalar dependencias de la API

### Opción A: Local (sin Docker para la API)

```bash
cd api
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cd ..
```

### Opción B: Docker (recomendada en este proyecto)

```bash
docker compose build api
```

## 4) Crear estructura estándar de FastAPI

Desde la raíz del repo:

```bash
mkdir -p api/app/api/v1/endpoints
mkdir -p api/app/core
mkdir -p api/app/schemas
mkdir -p api/app/services
mkdir -p api/app/models
mkdir -p api/app/clients
mkdir -p api/app/agents

touch api/app/api/__init__.py
touch api/app/__init__.py
touch api/app/api/v1/__init__.py
touch api/app/api/v1/endpoints/__init__.py
touch api/app/core/__init__.py
touch api/app/schemas/__init__.py
touch api/app/services/__init__.py
touch api/app/models/__init__.py
touch api/app/clients/__init__.py
touch api/app/agents/__init__.py
```

## 5) Crear código (archivo por archivo)

## 5.1 `api/requirements.txt`

```txt
fastapi==0.115.0
uvicorn[standard]==0.30.6
openai>=1.40.0,<2.0.0
langgraph>=0.2.0,<0.4.0
pydantic-settings>=2.2.1,<3.0.0
```

## 5.2 `api/app/core/settings.py`

```python
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "LLM Enterprise Platform API"
    app_description: str = "API base del curso LLMs Locales - Ollama, vLLM y sus Alternativas"
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
    return Settings()
```

## 5.3 `api/app/core/exceptions.py`

```python
class UpstreamServiceError(Exception):
    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(detail)
```

## 5.4 `api/app/clients/openai_client.py`

```python
from functools import lru_cache
from openai import OpenAI
from app.core.settings import get_settings


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    settings = get_settings()
    return OpenAI(
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        timeout=settings.openai_timeout_seconds,
    )
```

## 5.5 `api/app/models/chat.py`

```python
from dataclasses import dataclass


@dataclass(slots=True)
class ChatMessage:
    role: str
    content: str


@dataclass(slots=True)
class ChatRequest:
    model: str
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
```

## 5.6 `api/app/schemas/chat.py`

```python
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from app.models.chat import ChatMessage, ChatRequest


class ChatMessageSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ChatCompletionRequestSchema(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "model": "llama3.1:8b",
                "messages": [
                    {"role": "user", "content": "Explica LangGraph en una frase corta."}
                ],
                "temperature": 0.2,
                "max_tokens": 120,
                "stream": False,
            }
        },
    )
    model: str = Field(
        min_length=1,
        description="Modelo obligatorio que se usara en esta llamada.",
    )
    messages: list[ChatMessageSchema] = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int = Field(
        ge=1,
        description="Max tokens obligatorio para esta llamada.",
    )
    stream: Literal[False] = False

    def to_domain(self) -> ChatRequest:
        messages = [ChatMessage(role=m.role, content=m.content) for m in self.messages]
        return ChatRequest(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
```

## 5.7 `api/app/agents/chat_agent.py`

```python
from typing import Any, TypedDict
from langgraph.graph import END, StateGraph
from app.clients.openai_client import get_openai_client
from app.models.chat import ChatRequest


class ChatAgentState(TypedDict, total=False):
    model: str
    messages: list[dict[str, str]]
    temperature: float | None
    max_tokens: int
    completion: dict[str, Any]


def _call_llm(state: ChatAgentState) -> ChatAgentState:
    payload: dict[str, Any] = {
        "model": state["model"],
        "messages": state["messages"],
        "max_tokens": state["max_tokens"],
        "stream": False,
    }
    if state.get("temperature") is not None:
        payload["temperature"] = state["temperature"]

    completion = get_openai_client().chat.completions.create(**payload)
    return {"completion": completion.model_dump(mode="json")}


def _build_chat_agent():
    builder = StateGraph(ChatAgentState)
    builder.add_node("call_llm", _call_llm)
    builder.set_entry_point("call_llm")
    builder.add_edge("call_llm", END)
    return builder.compile()


_chat_agent = None


def initialize_chat_agent() -> None:
    global _chat_agent
    if _chat_agent is None:
        _chat_agent = _build_chat_agent()


def run_chat_agent(request: ChatRequest) -> dict[str, Any]:
    global _chat_agent
    if _chat_agent is None:
        initialize_chat_agent()

    state: ChatAgentState = {
        "model": request.model,
        "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
    }
    result = _chat_agent.invoke(state)
    return result["completion"]
```

## 5.8 `api/app/services/llm_service.py`

```python
from typing import Any
from openai import APIConnectionError, APIError, APITimeoutError
from app.agents.chat_agent import run_chat_agent
from app.clients.openai_client import get_openai_client
from app.core.exceptions import UpstreamServiceError
from app.models.chat import ChatRequest


def list_models() -> dict[str, Any]:
    client = get_openai_client()
    try:
        response = client.models.list()
    except (APIConnectionError, APITimeoutError, APIError) as exc:
        raise UpstreamServiceError(f"Error listing models from upstream provider: {exc}") from exc
    return response.model_dump(mode="json")


def create_chat_completion(request: ChatRequest) -> dict[str, Any]:
    try:
        return run_chat_agent(request)
    except (APIConnectionError, APITimeoutError, APIError) as exc:
        raise UpstreamServiceError(f"Error generating chat completion from upstream provider: {exc}") from exc
```

## 5.9 `api/app/api/v1/endpoints/base.py`

```python
from datetime import datetime, timezone
from fastapi import APIRouter
from app.core.settings import get_settings

router = APIRouter(tags=["Base"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/")
def root() -> dict[str, str]:
    settings = get_settings()
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "docs": f"{settings.app_root_path}/docs",
    }
```

## 5.10 `api/app/api/v1/endpoints/llm.py`

```python
from typing import Any
from fastapi import APIRouter
from app.schemas.chat import ChatCompletionRequestSchema
from app.services import llm_service

router = APIRouter(prefix="/v1", tags=["LLM"])


@router.get("/models")
def list_models() -> dict[str, Any]:
    return llm_service.list_models()


@router.post("/chat/completions")
def create_chat_completion(payload: ChatCompletionRequestSchema) -> dict[str, Any]:
    request = payload.to_domain()
    return llm_service.create_chat_completion(request)
```

## 5.11 `api/app/api/v1/router.py`

```python
from fastapi import APIRouter
from app.api.v1.endpoints import base, llm

router = APIRouter()
router.include_router(base.router)
router.include_router(llm.router)
```

## 5.12 `api/app/main.py`

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from app.agents.chat_agent import initialize_chat_agent
from app.api.v1.router import router as api_v1_router
from app.core.exceptions import UpstreamServiceError
from app.core.settings import get_settings

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
    initialize_chat_agent()


@app.exception_handler(UpstreamServiceError)
def handle_upstream_service_error(_request: Request, exc: UpstreamServiceError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": exc.detail})
```

## 6) Propagar variables `.env` al contenedor API (Docker Compose)

En `docker-compose.yml`, dentro del servicio `api`, añade:

```yaml
environment:
  - OPENAI_BASE_URL=${OPENAI_BASE_URL:-http://localhost:8001/v1}
  - OPENAI_API_KEY=${OPENAI_API_KEY:-dummy}
  - OPENAI_TIMEOUT_SECONDS=${OPENAI_TIMEOUT_SECONDS:-60}
```

## 7) Ejecutar la API

### Opción A: Local

```bash
cd api
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Opción B: Docker

```bash
docker compose build api
docker compose --profile vllm up -d
docker compose logs -f api
```

## 8) Probar endpoints

## 8.1 Health

```bash
curl http://localhost:8000/health
```

Esperado:

```json
{"status":"ok"}
```

## 8.2 Listar modelos

```bash
curl http://localhost:8000/v1/models
```

Esperado: JSON tipo OpenAI con campo `data`.

## 8.3 Chat completion (modelo y max_tokens obligatorios)

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.1:8b",
    "messages": [
      {"role": "user", "content": "Resume en una frase qué es LangGraph."}
    ],
    "temperature": 0.2,
    "max_tokens": 80
  }'
```

## 8.4 Chat completion inválido (falta `model` y `max_tokens`)

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Responde con OK"}
    ]
  }'
```

Esperado: `422` por campos obligatorios.

## 9) Casos de validación/errores para enseñar en clase

- `stream=true` debe fallar (solo soportamos no-stream en esta sesión):

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "hola"}],
    "stream": true
  }'
```

- `messages` vacío debe devolver `422`.
- `model` y `max_tokens` son obligatorios y deben enviarse siempre.
- `OPENAI_BASE_URL` inválida debe devolver `502` al llamar `/v1/models` o `/v1/chat/completions`.

## 10) Troubleshooting rápido

- Si `/v1/models` falla:
  - Verifica que vLLM está arriba:
    ```bash
    curl http://localhost:8001/v1/models
    ```
  - Revisa `OPENAI_BASE_URL` en `.env`.

- Si cambiaste dependencias y no se reflejan en Docker:
  ```bash
  docker compose build --no-cache api
  docker compose --profile vllm up -d
  ```

- Si no arranca FastAPI por imports:
  - Asegura que ejecutas desde `api/` cuando lanzas `uvicorn app.main:app`.
