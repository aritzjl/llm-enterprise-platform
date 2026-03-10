# Sesión 1 - Paso 2

Objetivo: añadir observabilidad de `POST /v1/chat/completions` con Langfuse Cloud, usando `environment` configurable desde `.env`.

## 1) Partir de la rama del paso 1

```bash
git switch sesion1/paso1
git switch -c sesion1/paso2
```

## 2) Instalar dependencia Langfuse

Editar `api/requirements.txt` y añadir:

```txt
langfuse>=2.59.3,<3.0.0
```

## 3) Configurar variables de entorno

Añadir en `.env`:

```dotenv
LANGFUSE_SECRET_KEY=sk-lf-ac8a3694-3306-43b7-9bd4-5f4cc85724fc
LANGFUSE_PUBLIC_KEY=pk-lf-0dede184-155c-4fdd-9190-9be310d09e0e
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_ENVIRONMENT=sesion1-paso2
```

Añadir lo mismo en `.env.example` para que el grupo tenga el mismo punto de partida.

## 4) Extender settings de FastAPI

En `api/app/core/settings.py` añadir:

```python
langfuse_secret_key: str = ""
langfuse_public_key: str = ""
langfuse_base_url: str = "https://cloud.langfuse.com"
langfuse_environment: str = "sesion1-paso2"
```

## 5) Crear capa de observabilidad

Crear carpeta y archivo:

```bash
mkdir -p api/app/observability
touch api/app/observability/__init__.py
touch api/app/observability/langfuse.py
```

Implementar en `api/app/observability/langfuse.py`:
- factoría singleton `get_langfuse_client()` con `Langfuse(public_key, secret_key, host)`
- helper `create_chat_trace(...)`
- helper `create_llm_generation(...)`
- helpers de cierre:
  - `update_chat_trace_success(...)`
  - `update_chat_trace_error(...)`
  - `end_llm_generation_success(...)`
  - `end_llm_generation_error(...)`
- `flush_langfuse()` para vaciar cola al apagar.

Notas:
- La traza debe incluir `environment=settings.langfuse_environment`.
- La generación también debe incluir `environment=settings.langfuse_environment`.
- Si Langfuse falla, solo loggear warning (best effort), sin romper la API.

## 6) Instrumentar servicio y agente

### 6.1 Servicio (`api/app/services/llm_service.py`)

- Crear trace al entrar en `create_chat_completion`.
- Medir latencia total.
- Pasar el trace al agente.
- Al final:
  - en éxito: `update_chat_trace_success(...)`
  - en error: `update_chat_trace_error(...)`.

### 6.2 Agente (`api/app/agents/chat_agent.py`)

- Extender `run_chat_agent(..., trace=None)`.
- En `_call_llm`:
  - crear generation hija con `trace.generation(...)`,
  - llamar al modelo OpenAI-compatible,
  - registrar `output`, `usage_details` y latencia,
  - cerrar generation en éxito/error.

## 7) Flush en shutdown

En `api/app/main.py`:
- importar `flush_langfuse`
- añadir hook:

```python
@app.on_event("shutdown")
def on_shutdown() -> None:
    flush_langfuse()
```

## 8) Pasar variables al contenedor API

En `docker-compose.yml`, servicio `api`, añadir:

```yaml
environment:
  - LANGFUSE_SECRET_KEY=${LANGFUSE_SECRET_KEY:-}
  - LANGFUSE_PUBLIC_KEY=${LANGFUSE_PUBLIC_KEY:-}
  - LANGFUSE_BASE_URL=${LANGFUSE_BASE_URL:-https://cloud.langfuse.com}
  - LANGFUSE_ENVIRONMENT=${LANGFUSE_ENVIRONMENT:-sesion1-paso2}
```

## 9) Ejecutar y verificar

```bash
docker compose build api
docker compose up -d api
```

Probar chat:

```bash
curl -X POST 'http://localhost:8000/curso/api/v1/chat/completions' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "llama3.1:8b",
    "messages": [{"role":"user","content":"Dime hola"}],
    "max_tokens": 32,
    "temperature": 0.2,
    "stream": false
  }'
```

Validar en Langfuse Cloud:
- entra en `https://cloud.langfuse.com`
- filtra por `environment=sesion1-paso2`
- confirma:
  - trace `chat.completions`
  - generation `llm.generation`
  - input/output
  - usage y latencia.

## 10) Comprobación de resiliencia

Cambiar temporalmente `LANGFUSE_SECRET_KEY` a un valor inválido y repetir el chat:
- la API debe seguir devolviendo respuesta.
- en logs debe aparecer warning de observabilidad.

## 11) Commit del paso 2

```bash
git add .
git commit -m "feat: sesion1 paso2 integra observabilidad con langfuse cloud"
```
