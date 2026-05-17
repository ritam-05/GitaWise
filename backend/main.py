"""FastAPI backend entrypoint for GitaWise."""

from __future__ import annotations

import logging
import os

# Configure logging FIRST before any other code
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logger.info("[BOOT] ========== GitaWise Backend Starting ==========")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger.info("[BOOT] Importing routes...")
from backend.routes.health import router as health_router
logger.info("[BOOT] ✓ Health router imported")
from backend.routes.query_engine import router as query_engine_router
logger.info("[BOOT] ✓ Query engine router imported")

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


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize resources at app startup.
    
    Models are now lazily loaded on first request to speed up development.
    """
    logger.info("=" * 60)
    logger.info("[STARTUP] GitaWise Backend Startup Event")
    logger.info("=" * 60)
    logger.info("[STARTUP] ✓ Models will be initialized on first request (lazy loading)")
    logger.info("[STARTUP] ✓ Backend ready for requests at http://127.0.0.1:8000")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Clean up resources during shutdown."""
    logger.info("GitaWise Backend shutting down...")

