"""FastAPI backend entrypoint for GitaWise."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.health import router as health_router
from backend.routes.query_engine import router as query_engine_router

app = FastAPI(
    title="GitaWise Backend",
    version="0.1.0",
    description="FastAPI backend for Bhagavad Gita query understanding and retrieval.",
)

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)

app.include_router(health_router)
app.include_router(query_engine_router)
