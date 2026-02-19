from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@router.get("/")
def root() -> dict[str, str]:
    """Informacion del servicio."""
    return {
        "service": "LLM Enterprise Platform API",
        "version": "0.1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "docs": "/curso/api/docs",
    }
