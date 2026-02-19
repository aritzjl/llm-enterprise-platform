"""
LLM Enterprise Platform - API Base
===================================
Servicio FastAPI base para el curso de LLMs Locales.
Este servicio se extiende con chat OpenAI-compatible para la sesion 1.
"""

from __future__ import annotations

from fastapi import FastAPI

from app.api.router import api_router

app = FastAPI(
    title="LLM Enterprise Platform - API",
    description="API base del curso LLMs Locales - Ollama, vLLM y sus Alternativas",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    root_path="/curso/api",
)

app.include_router(api_router)
