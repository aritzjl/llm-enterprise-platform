# Sesion 2 - Paso 4

Objetivo: introducir un modelo enrutador configurable por request (`router_model`) y refactorizar el agente LangGraph a una estructura modular profesional (`graph`, `nodes`, `prompts`, `state`) para decidir si responde el modelo pequeno o se enruta al modelo grande.

## 1) Crear rama del paso 4

```bash
git switch sesion1/paso3
git switch -c sesion2/paso4
```

## 2) Extender el contrato de chat

En `api/app/schemas/chat.py`:
- anadir `router_model` obligatorio en `ChatCompletionRequestSchema`.
- mantener `model` como el modelo principal (grande).
- actualizar `json_schema_extra` con un ejemplo que incluya `router_model`.
- en `to_domain()`, normalizar `router_model` con `.strip()`.

Ejemplo de payload:

```json
{
  "provider": "ollama",
  "base_url": "http://localhost:11436/v1",
  "router_model": "llama3.2:1b",
  "model": "llama3.1:8b",
  "messages": [{"role": "user", "content": "Dime hola en una frase"}],
  "max_tokens": 64,
  "temperature": 0.2,
  "stream": false
}
```

## 3) Extender el modelo de dominio

En `api/app/models/chat.py`, actualizar `ChatRequest` para incluir:
- `router_model: str` (obligatorio).

## 4) Refactor del agente a paquete modular

Eliminar el modulo antiguo `api/app/agents/chat_agent.py` y crear el paquete:

```text
api/app/agents/chat_completion_agent/
  __init__.py
  graph.py
  nodes.py
  prompts.py
  state.py
```

### 4.1 `state.py`

Definir:
- `ChatAgentRoute = Literal["simple", "complex"]`
- `ChatAgentState` con `provider`, `base_url`, `router_model`, `model`, `messages`, `selected_route`, `responder_model`, `completion`, `trace`.
- `ChatAgentResult` publico con `completion`, `selected_route`, `responder_model`.

### 4.2 `prompts.py`

Crear prompt de clasificacion con salida JSON estricta:
- solo permitir `{"route":"simple"}` o `{"route":"complex"}`.
- sin markdown, sin texto adicional.

### 4.3 `nodes.py`

Crear nodos puros:
- `classify_route_node`: llama `router_model` para clasificar.
- `call_router_model_node`: si ruta `simple`, genera respuesta final con `router_model`.
- `call_primary_model_node`: si ruta `complex`, genera respuesta final con `model`.
- `route_selector`: fallback seguro a `complex` ante valores invalidos.

Notas importantes:
- parseo robusto de JSON de clasificacion (incluyendo limpieza de fences ```json).
- fallback a `complex` ante salida no parseable.
- observabilidad por generacion:
  - `llm.router.classification`
  - `llm.router.response`
  - `llm.generation`

### 4.4 `graph.py`

Construir `StateGraph` con flujo:
- `classify_route` -> edge condicional -> `call_router_model` o `call_primary_model` -> `END`.

Exponer:
- `initialize_chat_agent()` (compilacion singleton)
- `run_chat_agent(request, trace=None)` devolviendo `ChatAgentResult`.

### 4.5 `__init__.py`

Reexportar API publica del paquete:
- `initialize_chat_agent`
- `run_chat_agent`

## 5) Migrar imports al nuevo agente

Actualizar:
- `api/app/main.py` -> importar `initialize_chat_agent` desde `app.agents.chat_completion_agent`.
- `api/app/services/llm_service.py` -> importar `run_chat_agent` desde `app.agents.chat_completion_agent`.
- `api/app/agents/__init__.py` -> reexportar desde `chat_completion_agent`.

## 6) Extender observabilidad de Langfuse

En `api/app/observability/langfuse.py`:

### 6.1 Trace de chat

En `create_chat_trace(...)`, incluir `router_model` en:
- `trace.input`
- `trace.metadata`

### 6.2 Update de trace

Extender helpers:
- `update_chat_trace_success(...)`
- `update_chat_trace_error(...)`

Nuevos campos:
- `router_model` (obligatorio en helper)
- `selected_route` (opcional)
- `responder_model` (opcional)

### 6.3 Generaciones

Extender `create_llm_generation(...)` para permitir:
- `generation_name` configurable
- `metadata` extra

Extender cierre de generacion para metadata adicional:
- `end_llm_generation_success(..., extra_metadata=...)`
- `end_llm_generation_error(..., extra_metadata=...)`

## 7) Ajustar servicio de chat

En `api/app/services/llm_service.py`:
- consumir `ChatAgentResult` del agente (`completion`, `selected_route`, `responder_model`).
- devolver solo `completion` al cliente.
- al actualizar trace de exito/error, propagar:
  - `provider`
  - `base_url`
  - `router_model`
  - `selected_route`
  - `responder_model`

## 8) Ejecutar

```bash
docker compose build api
docker compose up -d api
```

## 9) Pruebas

### 9.1 Ruta simple (responde router)

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

Esperado en Langfuse:
- `selected_route = simple`
- `responder_model = router_model`

### 9.2 Ruta compleja (responde modelo grande)

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

Esperado en Langfuse:
- `selected_route = complex`
- `responder_model = model`

### 9.3 Fallback de clasificacion

Forzar temporalmente prompt de clasificacion para devolver texto no JSON y repetir una llamada:
- debe enrutar a `complex` por seguridad.
- en metadata del trace/generation debe quedar reflejado `selected_route = complex`.

### 9.4 Regresion de modelos

`GET /v1/models` debe seguir funcionando igual:

```bash
curl 'http://localhost:8000/curso/api/v1/models?provider=ollama&base_url=http://localhost:11436/v1'
```

## 10) Validacion en Langfuse

En `https://cloud.langfuse.com`:
- filtrar por el `environment` configurado en `.env`.
- validar en trace metadata/input:
  - `provider`
  - `base_url`
  - `router_model`
  - `selected_route`
  - `responder_model`
- validar generaciones:
  - `llm.router.classification`
  - `llm.router.response` (si ruta simple)
  - `llm.generation` (si ruta complex)

## 11) Commit

```bash
git add .
git commit -m "feat: sesion2 paso4 router model configurable y agente langgraph modular"
```
