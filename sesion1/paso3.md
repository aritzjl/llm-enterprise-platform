# Sesión 1 - Paso 3

Objetivo: comparar vLLM vs Ollama enviando `provider` y `base_url` en cada llamada a `POST /v1/chat/completions` y `GET /v1/models`, y guardar esos datos en metadata de Langfuse (chat).

## 1) Crear rama del paso 3

```bash
git switch sesion1/paso2
git switch -c sesion1/paso3
```

## 2) Cambiar contrato del endpoint de chat

En `api/app/schemas/chat.py`, añadir campos obligatorios:
- `provider`: `"vllm"` o `"ollama"`
- `base_url`: URL OpenAI-compatible por llamada.

Ejemplo de payload:

```json
{
  "provider": "ollama",
  "base_url": "http://localhost:11436/v1",
  "model": "llama3.1:8b",
  "messages": [{"role": "user", "content": "Dime hola"}],
  "max_tokens": 64,
  "temperature": 0.2,
  "stream": false
}
```

## 3) Extender el modelo de dominio

En `api/app/models/chat.py` incorporar en `ChatRequest`:
- `provider: str`
- `base_url: str`

## 3.1) Ajustar endpoint de modelos

En `GET /v1/models`:
- hacer `base_url` obligatorio como query param.
- aceptar `provider` opcional como query param para consistencia en la comparativa.

Ejemplo:

```bash
curl 'http://localhost:8000/curso/api/v1/models?provider=ollama&base_url=http://localhost:11436/v1'
```

## 4) Cliente OpenAI por base URL dinámica

En `api/app/clients/openai_client.py`:
- cambiar a factoría cacheada por `base_url` (`_build_openai_client(...)`)
- `get_openai_client(base_url=...)` debe crear/reusar cliente según la URL recibida en API.

Con esto, una request puede ir contra vLLM y la siguiente contra Ollama sin reiniciar API.

## 5) Pasar provider/base_url por el agente

En `api/app/agents/chat_agent.py`:
- incluir `provider` y `base_url` en el estado del grafo.
- usar `get_openai_client(base_url=state["base_url"])` al llamar `chat.completions.create`.

## 6) Guardar provider y base_url en Langfuse

En `api/app/observability/langfuse.py`:
- incluir `provider` + `base_url`:
  - en `trace.input`
  - en `trace.metadata`
  - en `generation.metadata`
- en updates de éxito/error del trace, volver a incluirlos para facilitar filtros posteriores.

## 7) Servicio de chat

En `api/app/services/llm_service.py`, al actualizar trace (éxito/error), pasar:
- `provider=request.provider`
- `base_url=request.base_url`

## 8) Ejecutar

```bash
docker compose build api
docker compose up -d api
```

## 9) Pruebas de comparativa

### 9.1 Listado de modelos por proveedor

```bash
curl 'http://localhost:8000/curso/api/v1/models?provider=vllm&base_url=http://localhost:8001/v1'
curl 'http://localhost:8000/curso/api/v1/models?provider=ollama&base_url=http://localhost:11436/v1'
```

### 9.2 Chat con vLLM

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

### 9.3 Chat con Ollama

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

## 10) Validación en Langfuse

En `https://cloud.langfuse.com`:
- filtra por `environment=sesion1-paso2` (o el valor que uses en `.env`)
- revisa metadata del trace/generation y confirma:
  - `provider` (`vllm`/`ollama`)
  - `base_url` usado en esa llamada.

## 11) Commit

```bash
git add .
git commit -m "feat: sesion1 paso3 comparativa vllm vs ollama con provider/base_url por request"
```
