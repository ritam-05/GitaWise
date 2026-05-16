"""Health and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from backend.routes.query_engine import get_query_engine

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Simple health-check payload."""

    status: str
    engine_ready: bool


@router.get("/", response_model=HealthResponse)
def root() -> HealthResponse:
    """Basic API status endpoint."""
    try:
        get_query_engine()
        return HealthResponse(status="ok", engine_ready=True)
    except Exception:
        return HealthResponse(status="degraded", engine_ready=False)


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Detailed readiness endpoint."""
    try:
        get_query_engine()
        return HealthResponse(status="ok", engine_ready=True)
    except Exception:
        return HealthResponse(status="degraded", engine_ready=False)
