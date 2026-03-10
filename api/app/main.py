"""FastAPI bootstrap for the LLM Enterprise Platform API."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.agents.chat_agent import initialize_chat_agent
from app.api.v1.router import router as api_v1_router
from app.core.exceptions import UpstreamServiceError
from app.core.settings import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    root_path=settings.app_root_path,
)
app.include_router(api_v1_router)


@app.on_event("startup")
def on_startup() -> None:
    """Initialize long-lived app components."""
    initialize_chat_agent()


@app.exception_handler(UpstreamServiceError)
def handle_upstream_service_error(
    _request: Request, exc: UpstreamServiceError
) -> JSONResponse:
    """Map upstream provider errors to a consistent API response."""
    return JSONResponse(status_code=502, content={"detail": exc.detail})
