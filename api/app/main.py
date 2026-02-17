"""
LLM Enterprise Platform - API Base
===================================
Servicio FastAPI base para el curso de LLMs Locales.
Este servicio se extendera durante las sesiones del curso.
"""

from datetime import datetime, timezone

from fastapi import FastAPI

app = FastAPI(
    title="LLM Enterprise Platform - API",
    description="API base del curso LLMs Locales - Ollama, vLLM y sus Alternativas",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    root_path="/curso/api",
)


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/")
def root():
    """Informacion del servicio."""
    return {
        "service": "LLM Enterprise Platform API",
        "version": "0.1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "docs": "/curso/api/docs",
    }
