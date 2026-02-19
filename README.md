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
- (Opcional) **Ollama** como motor de inferencia alternativo

Todos los servicios accesibles mediante puertos directos (sin gateway).

## Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                       Docker Network (llm-network)              │
│                                                                 │
│  FastAPI      vLLM       Qdrant     Langfuse      Ollama       │
│  (API Base)  (Inference) (VectorDB) (Observ.)    (perfil)      │
│   :8000       :8001       :6333      :3001        :11436       │
│                                        │                        │
│                         ┌──────────────┼──────────┐            │
│                         │              │          │            │
│                      Postgres    ClickHouse    Redis           │
│                       :5432        :8123       :6379           │
│                                                  │              │
│                                                MinIO            │
│                                                :9091            │
└─────────────────────────────────────────────────────────────────┘
```

**Puertos expuestos al host:**
- **FastAPI**: 8000
- **vLLM**: 8001
- **Qdrant**: 6333-6334
- **Langfuse**: 3001
- **Ollama**: 11436 (solo con perfil)
- **MinIO**: 9091

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

### Verificar GPU (solo Linux con NVIDIA)

```bash
nvidia-smi                          # Drivers NVIDIA

# Verificar nvidia-container-toolkit
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

### Configurar contexto de Docker

**IMPORTANTE**: Para que vLLM funcione con GPU, debes usar **Docker Engine** (contexto `default`), no Docker Desktop:

```bash
# Ver contexto actual
docker context show

# Si no es "default", cambiar a Docker Engine
docker context use default
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
# IMPORTANTE: Asegurarse de usar Docker Engine (no Docker Desktop)
docker context use default

# Stack base con vLLM (requiere NVIDIA GPU)
docker compose --profile vllm up -d

# Stack base con Ollama (motor alternativo, CPU/GPU)
docker compose --profile ollama up -d

# Stack completo (vLLM + Ollama)
docker compose --profile vllm --profile ollama up -d
```

**Nota sobre vLLM y GPU:**
- vLLM usa la versión `v0.5.4` compatible con CUDA 12.9
- Requiere GPU NVIDIA con al menos 2 GB VRAM
- Si no tienes GPU, usa el perfil Ollama que funciona en CPU

### 4. Esperar a que los servicios arranquen

Langfuse tarda 2-3 minutos en estar listo. vLLM necesita descargar el modelo la primera vez (~1 GB).

```bash
# Ver el estado de los contenedores
docker compose ps

# Ver logs en tiempo real
docker compose logs -f
```

## Rutas Disponibles

Todos los servicios están accesibles mediante puertos directos:

| Servicio | URL | Notas |
|----------|-----|-------|
| **API - Health** | http://localhost:8000/health | Health check |
| **API - Docs** | http://localhost:8000/docs | Swagger UI |
| **vLLM - Modelos** | http://localhost:8001/v1/models | Lista de modelos |
| **vLLM - Chat** | http://localhost:8001/v1/chat/completions | API OpenAI-compatible |
| **Ollama - Tags** | http://localhost:11436/api/tags | Lista de modelos (solo perfil ollama) |
| **Ollama - Generate** | http://localhost:11436/api/generate | Generación de texto (solo perfil ollama) |
| **Qdrant - Dashboard** | http://localhost:6333/dashboard | UI web |
| **Qdrant - Health** | http://localhost:6333/healthz | Health check |
| **Langfuse** | http://localhost:3001 | UI de observabilidad |
| **MinIO** | http://localhost:9091 | Almacenamiento S3-compatible |

## Checks Obligatorios

Antes de la sesion 1, el alumno debe verificar que todos los servicios responden correctamente.

### 1. FastAPI

```bash
curl http://localhost:8000/health
# Respuesta esperada: {"status":"ok"}
```

### 2. vLLM

```bash
# Listar modelos
curl http://localhost:8001/v1/models

# Test de inferencia
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2-1.5B-Instruct-AWQ",
    "messages": [{"role": "user", "content": "Hola, responde en una frase corta."}],
    "max_tokens": 50
  }'
```

### 3. Qdrant

```bash
curl http://localhost:6333/healthz
# Respuesta esperada: (vacio o "healthz check passed")
```

### 4. Langfuse

Abrir en el navegador: http://localhost:3001

Deberia aparecer la pantalla de registro/login de Langfuse.

### 5. Ollama (si se activo el perfil)

```bash
# Listar modelos
curl http://localhost:11436/api/tags

# Test de generación
curl http://localhost:11436/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2:1.5b",
    "prompt": "Hola, responde en una frase corta.",
    "stream": false
  }'
```

## Modelos de Inferencia

| Motor | Modelo | Tamaño | Contexto | GPU |
|-------|--------|--------|----------|-----|
| **vLLM v0.5.4** | `Qwen/Qwen2-1.5B-Instruct-AWQ` | ~1 GB (AWQ 4-bit) | 8192 tokens | Requerida (NVIDIA) |
| **Ollama** | `qwen2:1.5b` | ~935 MB (GGUF Q4_0) | 32K tokens | Opcional (funciona en CPU) |

Ambos modelos son variantes cuantizadas del Qwen2-1.5B-Instruct, optimizadas para inferencia eficiente.

## Estructura del Repositorio

```
llm-enterprise-platform/
├── .env.example              # Template de variables de entorno
├── .gitignore
├── docker-compose.yml        # Definicion de todos los servicios
├── README.md
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
# Levantar stack con vLLM (GPU)
docker compose --profile vllm up -d

# Levantar stack con Ollama (CPU/GPU)
docker compose --profile ollama up -d

# Levantar ambos
docker compose --profile vllm --profile ollama up -d

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

**Síntoma:** El contenedor `llm-vllm` se para inmediatamente o muestra error "could not select device driver nvidia".

```bash
docker logs llm-vllm
```

**Causas comunes:**
1. **Contexto de Docker incorrecto** (usando Docker Desktop en vez de Docker Engine)
2. No hay GPU NVIDIA disponible
3. Drivers NVIDIA no instalados o incompatibles
4. `nvidia-container-toolkit` no instalado
5. Memoria GPU insuficiente (necesita ~2 GB VRAM)

**Soluciones:**

1. **Verificar y cambiar contexto de Docker:**
```bash
docker context show                    # Debe ser "default"
docker context use default             # Cambiar a Docker Engine
sudo systemctl restart docker          # Reiniciar Docker
```

2. **Verificar GPU y drivers:**
```bash
nvidia-smi                            # Debe mostrar tu GPU
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

3. **Si el error persiste con "Error 803: unsupported display driver / cuda driver combination":**
   - La versión de vLLM está configurada en `v0.5.4` que es compatible con CUDA 12.9
   - Verifica que tus drivers NVIDIA soporten CUDA 12.9 (`nvidia-smi` muestra la versión)

4. **Alternativa sin GPU:** Usa Ollama en su lugar:
```bash
docker compose down
docker compose --profile ollama up -d
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

### Conflicto de puertos

Si algún puerto está ocupado:

```bash
# Verificar qué proceso usa un puerto (ejemplo: 8000)
ss -tuln | grep 8000
sudo lsof -i :8000

# Solución: cambiar el puerto en docker-compose.yml
# Ejemplo: cambiar "8000:8000" por "8080:8000"
```

**Puertos comunes que pueden estar ocupados:**
- **11434** - Ollama del sistema host (cambiar a 11436 en docker-compose.yml)
- **3000** - Otras aplicaciones Node.js (cambiar a 3001)
- **8000** - Otras APIs (cambiar a 8080)
- **9090** - Prometheus u otros servicios (cambiar a 9091)

### Reset completo

Si algo no funciona despues de intentar varias cosas:
```bash
# Ejemplo para Ollama (reemplazar perfil si se usa vLLM)
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
