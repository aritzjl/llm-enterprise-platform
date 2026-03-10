"""Root router for v1 endpoints."""

from fastapi import APIRouter

from app.api.v1.endpoints import base, llm

router = APIRouter()
router.include_router(base.router)
router.include_router(llm.router)
