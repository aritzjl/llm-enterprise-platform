# LLMs Locales - Prework Infraestructura Base

Infraestructura base obligatoria para el curso **LLMs Locales - Ollama, vLLM y sus Alternativas**.

Este repositorio **no contiene** la logica del proyecto final ni los ejercicios del curso.
Unicamente proporciona la base sobre la que construiremos la plataforma LLM corporativa durante las sesiones.

## Objetivo del Prework

Al finalizar el prework, cada alumno debera tener funcionando en su maquina:

- Un servidor de inferencia con **vLLM** (modelo `Qwen/Qwen2-1.5B-Instruct-AWQ`)
- Una base de datos vectorial con **Qdrant**
- Una API base con **FastAPI**
- Una plataforma de observabilidad LLM con **Langfuse**
- Un gateway comun con **Nginx**

Todo accesible desde: **http://localhost/curso/**

## Arquitectura

```
                    ┌──────────────┐
                    │    NGINX     │
                    │  (Gateway)   │
                    │   :80        │
                    └──────┬───────┘
                           │
  ┌────────────┬───────────┼───────────┬────────────────┐
  │            │           │           │                │
  ▼            ▼           ▼           ▼                ▼
FastAPI      vLLM       Qdrant     Langfuse         Ollama
(API Base)  (Inference) (VectorDB) (Observability)  (perfil)
 :8000       :8000       :6333      :3000           :11434
                                      │
                          ┌───────────┼───────────┐
                          │           │           │
                       Postgres   ClickHouse    Redis
                        :5432       :8123       :6379
                                                  │
                                                MinIO
                                                :9000
```

Cada servicio corre en su propio contenedor Docker dentro de una red compartida (`llm-network`).

## Requisitos Previos

| Requisito | Minimo | Notas |
|-----------|--------|-------|
| **Docker** | v24+ | Con Docker Compose v2 |
| **RAM** | 16 GB | Recomendado 24 GB+ |
| **GPU** | NVIDIA con drivers + `nvidia-container-toolkit` | Necesario para vLLM |
| **Disco** | 20 GB libres | Para imagenes Docker y modelos |
| **SO** | Linux (recomendado) / macOS / Windows (WSL2) | vLLM solo funciona con NVIDIA GPU |

### Verificar Docker

```bash
docker --version          # v24+
docker compose version    # v2+
```

### Verificar GPU (solo Linux)

```bash
nvidia-smi                          # Drivers NVIDIA
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi  # nvidia-container-toolkit
```

## Instalacion

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd llm-enterprise-platform
```

### 2. Configurar variables de entorno

```bash
cp .env.example .env
```

Los valores por defecto son funcionales para desarrollo local. No es necesario cambiar nada para el prework.

### 3. Levantar el stack

```bash
# Stack base (con vLLM - requiere NVIDIA GPU)
docker compose up -d

# Stack base + Ollama (para la comparativa del curso)
docker compose --profile ollama up -d
```

### 4. Esperar a que los servicios arranquen

Langfuse tarda 2-3 minutos en estar listo. vLLM necesita descargar el modelo la primera vez (~1 GB).

```bash
# Ver el estado de los contenedores
docker compose ps

# Ver logs en tiempo real
docker compose logs -f
```

## Rutas Disponibles

Todos los servicios estan accesibles desde el gateway Nginx bajo el prefijo `/curso/`.

| Servicio | URL | Notas |
|----------|-----|-------|
| **API - Health** | http://localhost/curso/api/health | Health check |
| **API - Docs** | http://localhost/curso/api/docs | Swagger UI |
| **vLLM - Modelos** | http://localhost/curso/vllm/v1/models | Lista de modelos |
| **vLLM - Chat** | http://localhost/curso/vllm/v1/chat/completions | API OpenAI-compatible |
| **Ollama - Chat** | http://localhost/curso/ollama/chat/completions | API OpenAI-compatible (perfil ollama) |
| **Ollama - Modelos** | http://localhost/curso/ollama/models | Lista de modelos (perfil ollama) |
| **Qdrant - Dashboard** | http://localhost/curso/qdrant/dashboard | UI web |
| **Qdrant - Health** | http://localhost/curso/qdrant/healthz | Health check |
| **Langfuse** | http://localhost/curso/langfuse/ | UI de observabilidad |
| **Langfuse (directo)** | http://localhost:3000 | Acceso directo sin Nginx |

## Checks Obligatorios

Antes de la sesion 1, el alumno debe verificar que todos los servicios responden correctamente.

### 1. FastAPI

```bash
curl http://localhost/curso/api/health
# Respuesta esperada: {"status":"ok"}
```

### 2. vLLM

```bash
# Listar modelos
curl http://localhost/curso/vllm/v1/models

# Test de inferencia
curl http://localhost/curso/vllm/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2-1.5B-Instruct-AWQ",
    "messages": [{"role": "user", "content": "Hola, responde en una frase corta."}],
    "max_tokens": 50
  }'
```

### 3. Qdrant

```bash
curl http://localhost/curso/qdrant/healthz
# Respuesta esperada: (vacio o "healthz check passed")
```

### 4. Langfuse

Abrir en el navegador: http://localhost:3000

Deberia aparecer la pantalla de registro/login de Langfuse.

### 5. Ollama (si se activo el perfil)

```bash
# Listar modelos
curl http://localhost/curso/ollama/models

# Test de inferencia
curl http://localhost/curso/ollama/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2:1.5b",
    "messages": [{"role": "user", "content": "Hola, responde en una frase corta."}],
    "max_tokens": 50
  }'
```

## Modelo de Inferencia

| Motor | Modelo | Tamano | Contexto |
|-------|--------|--------|----------|
| **vLLM** | `Qwen/Qwen2-1.5B-Instruct-AWQ` | ~1 GB (AWQ 4-bit) | 8192 tokens |
| **Ollama** | `qwen2:1.5b` | ~935 MB (GGUF Q4) | 32K tokens |

Ambos modelos son variantes cuantizadas del Qwen2-1.5B-Instruct, optimizadas para inferencia eficiente.

## Estructura del Repositorio

```
llm-enterprise-platform/
├── .env.example              # Template de variables de entorno
├── .gitignore
├── docker-compose.yml        # Definicion de todos los servicios
├── README.md
├── nginx/
│   └── default.conf          # Configuracion del gateway
├── api/
│   ├── Dockerfile            # Imagen de la API
│   ├── requirements.txt      # Dependencias Python
│   └── app/
│       └── main.py           # Aplicacion FastAPI
└── ollama/
    └── entrypoint.sh         # Script de arranque con auto-pull del modelo
```

## Comandos Utiles

```bash
# Levantar stack base
docker compose up -d

# Levantar stack con Ollama
docker compose --profile ollama up -d

# Ver estado de los contenedores
docker compose ps

# Ver logs de un servicio
docker compose logs -f vllm
docker compose logs -f langfuse-web

# Reiniciar un servicio
docker compose restart api

# Parar todo
docker compose down

# Parar todo y borrar volumenes (reset completo)
docker compose down -v

# Reconstruir la API tras cambios en el codigo
docker compose build api
docker compose up -d api
```

## Troubleshooting

### vLLM no arranca

**Sintoma:** El contenedor `llm-vllm` se para inmediatamente.

```bash
docker logs llm-vllm
```

**Causas comunes:**
- No hay GPU NVIDIA disponible
- Drivers NVIDIA no instalados
- `nvidia-container-toolkit` no instalado
- Memoria GPU insuficiente (necesita ~2 GB VRAM)

**Solucion:** Verificar que `nvidia-smi` funciona y que el toolkit esta instalado:
```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Langfuse no carga

**Sintoma:** La UI no aparece o da error 502.

Langfuse v3 tarda 2-3 minutos en arrancar completamente. Espera y verifica:
```bash
docker compose logs -f langfuse-web
# Buscar: "Ready" en los logs
```

Si persiste, verifica que sus dependencias estan healthy:
```bash
docker compose ps  # Todos los servicios de Langfuse deben estar "healthy" o "running"
```

### Puerto 80 ocupado

```bash
# Verificar que proceso usa el puerto 80
sudo lsof -i :80

# Opcion: cambiar el puerto en docker-compose.yml
# Cambiar "80:80" por "8080:80" en el servicio nginx
```

### Reset completo

Si algo no funciona despues de intentar varias cosas:
```bash
docker compose --profile ollama down -v
docker compose --profile ollama up -d
```

Esto borra todos los volumenes (datos) y empieza de cero.

## Que NO Incluye Este Repositorio

- Logica de RAG
- Integracion con LangGraph
- Evaluacion automatica
- Multi-modelo
- CRM fake
- Seguridad avanzada
- Codigo de sesiones

Todo eso se desarrollara progresivamente durante el curso, partiendo de esta base estable.

## Lo Que Construiremos

Durante el curso transformaremos esta base en:

- Una plataforma LLM corporativa completa
- RAG agentico avanzado
- Evaluacion automatica con LLM-as-judge
- Multi-tenant
- Integracion con CRM
- Optimizacion y escalabilidad
- Arquitectura enterprise final
