"""Base endpoints for health and service metadata."""

from datetime import datetime, timezone

from fastapi import APIRouter

from app.core.settings import get_settings

router = APIRouter(tags=["Base"])


@router.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@router.get("/")
def root() -> dict[str, str]:
    """Service metadata endpoint."""
    settings = get_settings()
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "docs": f"{settings.app_root_path}/docs",
    }
