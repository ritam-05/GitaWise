"""FastAPI backend entrypoint for GitaWise."""

from __future__ import annotations

from fastapi import FastAPI

from backend.routes.health import router as health_router
from backend.routes.query_engine import router as query_engine_router

app = FastAPI(
    title="GitaWise Backend",
    version="0.1.0",
    description="FastAPI backend for Bhagavad Gita query understanding and retrieval.",
)

app.include_router(health_router)
app.include_router(query_engine_router)
