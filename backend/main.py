"""FastAPI backend entrypoint for GitaWise."""

from __future__ import annotations

import logging
import os
import sys
import uuid

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SUPABASE_CACHE_TABLE, SUPABASE_SERVICE_KEY, SUPABASE_URL

# Configure logging FIRST before any other code
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logger.info("[BOOT] ========== GitaWise Backend Starting ==========")

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

logger.info("[BOOT] Importing cache layer...")
try:
    from backend.cache import CacheManager
    CACHE_AVAILABLE = True
    logger.info("[BOOT] ✓ Cache layer available")
except ImportError:
    CACHE_AVAILABLE = False
    logger.warning("[BOOT] Cache layer not available")

logger.info("[BOOT] Importing routes...")
from backend.routes.health import router as health_router
logger.info("[BOOT] ✓ Health router imported")
from backend.routes.query_engine import router as query_engine_router
logger.info("[BOOT] ✓ Query engine router imported")


class SessionMiddleware(BaseHTTPMiddleware):
    """Middleware to handle session_id for each request."""

    async def dispatch(self, request: Request, call_next) -> object:
        """Add or retrieve session_id."""
        # Try to get session_id from header
        session_id = request.headers.get("X-Session-ID")
        
        # Fall back to query parameter
        if not session_id:
            session_id = request.query_params.get("session_id")
        
        # Generate new one if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Store in request state
        request.state.session_id = session_id
        
        response = await call_next(request)
        # Return session_id in response header
        response.headers["X-Session-ID"] = session_id
        return response


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

# Add session middleware
app.add_middleware(SessionMiddleware)

app.include_router(health_router)
app.include_router(query_engine_router)


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize resources at app startup.
    
    - Loads models once (lazy loading deferred to first request)
    - Initializes global cache if available
    """
    logger.info("=" * 60)
    logger.info("[STARTUP] GitaWise Backend Startup Event")
    logger.info("=" * 60)
    
    # Initialize global cache
    if CACHE_AVAILABLE:
        try:
            cache_manager = CacheManager(
                max_memory_items=1000,
                default_ttl_seconds=3600,
                supabase_url=SUPABASE_URL,
                supabase_key=SUPABASE_SERVICE_KEY,
                supabase_table=SUPABASE_CACHE_TABLE,
            )
            app.state.cache_manager = cache_manager
            logger.info("[STARTUP] ✓ Global cache initialized (session TTL=1h)")
        except Exception as exc:
            logger.warning("[STARTUP] Failed to initialize cache: %s", exc)
            app.state.cache_manager = None
    else:
        app.state.cache_manager = None
        logger.warning("[STARTUP] Cache not available")
    
    logger.info("[STARTUP] ✓ Models will be initialized on first request (lazy loading)")
    logger.info("[STARTUP] ✓ Session-aware routing enabled")
    logger.info("[STARTUP] ✓ Backend ready for requests at http://127.0.0.1:8000")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Clean up resources during shutdown."""
    logger.info("GitaWise Backend shutting down...")

