"""Query engine routes for debugging and retrieval testing."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
logger.info("[ROUTES] Query engine routes module starting...")

from backend.query_engine import AdaptiveGitaEngine, GitaQueryEngine

logger.info("[ROUTES] ✓ Query engine modules imported")

router = APIRouter(prefix="/query-engine", tags=["query-engine"])


class QueryEngineRequest(BaseModel):
    """Request payload for query-engine endpoints."""

    query: str = Field(..., min_length=3, description="User query to analyze and retrieve against.")
    session_id: Optional[str] = Field(None, description="Optional session ID for cache scoping.")


class QueryAnalysisResponse(BaseModel):
    """Intermediate LLM-visible stages for debugging query understanding."""

    original_query: str
    query_chunks: list[str]
    llm_extracted_emotions: list[dict[str, str]]
    retrieval_queries: list[str]


class QueryRouteResponse(BaseModel):
    """Adaptive route selected for the incoming query."""

    original_query: str
    route: str


class EmotionNormalizationRequest(BaseModel):
    """Request payload for raw emotion normalization."""

    emotion: str = Field(..., min_length=1, description="Raw emotion to normalize.")


class EmotionNormalizationResponse(BaseModel):
    """Canonical emotion mapping response."""

    input_emotion: str
    mapped_emotions: list[str]


class GeneratedAnswerResponse(BaseModel):
    """Final grounded answer returned by the generation layer."""

    original_query: str
    route: str
    answer: str
    cited_verses: list[str]
    warnings: list[str]
    contexts: list[dict[str, object]]
    used_rag: bool


@lru_cache(maxsize=1)
def get_query_engine(cache_manager: Optional[object] = None) -> GitaQueryEngine:
    """Create and cache the query engine lazily for API use."""
    return GitaQueryEngine(cache_manager=cache_manager)


@lru_cache(maxsize=1)
def get_adaptive_engine(cache_manager: Optional[object] = None) -> AdaptiveGitaEngine:
    """Create and cache the adaptive route-aware engine for final answers."""
    return AdaptiveGitaEngine(cache_manager=cache_manager)


@router.post("/analyze", response_model=QueryAnalysisResponse)
def analyze_query(payload: QueryEngineRequest) -> QueryAnalysisResponse:
    """Show how the LLM decomposes the query and extracts emotions."""
    try:
        engine = get_query_engine()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize query engine: {exc}",
        ) from exc

    try:
        problems = engine.decompose_query(payload.query)
        emotions = engine.detect_emotions(problems)
        retrieval_queries = engine.build_queries(emotions)

        return QueryAnalysisResponse(
            original_query=payload.query,
            query_chunks=[item.problem for item in problems],
            llm_extracted_emotions=[
                {"problem": item.problem, "emotion": item.emotion} for item in emotions
            ],
            retrieval_queries=[item.query for item in retrieval_queries],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Query analysis failed: {exc}",
        ) from exc


@router.post("/route", response_model=QueryRouteResponse)
def route_query(payload: QueryEngineRequest) -> QueryRouteResponse:
    """Route a query to the most suitable handling pipeline."""
    try:
        engine = get_query_engine()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize query engine: {exc}",
        ) from exc

    try:
        result = engine.route_query(payload.query)
        return QueryRouteResponse(
            original_query=payload.query,
            route=result.route,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Query routing failed: {exc}",
        ) from exc


@router.post("/normalize-emotion", response_model=EmotionNormalizationResponse)
def normalize_emotion(payload: EmotionNormalizationRequest) -> EmotionNormalizationResponse:
    """Normalize a raw user emotion into canonical GitaWise emotion labels."""
    try:
        engine = get_query_engine()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize query engine: {exc}",
        ) from exc

    try:
        result = engine.normalize_emotion(payload.emotion)
        return EmotionNormalizationResponse(
            input_emotion=result.input_emotion,
            mapped_emotions=result.mapped_emotions,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Emotion normalization failed: {exc}",
        ) from exc


@router.post("/run")
def run_query_engine(request: Request, payload: QueryEngineRequest) -> dict:
    """Execute the full query-engine pipeline and return grounded context."""
    try:
        cache_manager = getattr(request.app.state, "cache_manager", None)
        engine = get_query_engine(cache_manager=cache_manager)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize query engine: {exc}",
        ) from exc

    try:
        session_id = payload.session_id or getattr(request.state, "session_id", None)
        if session_id and engine.session_cache:
            response = engine.run_with_session(payload.query, session_id)
        else:
            response = engine.run(payload.query)
        return response.model_dump()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Query engine execution failed: {exc}",
        ) from exc


@router.post("/answer", response_model=GeneratedAnswerResponse)
def answer_query(request: Request, payload: QueryEngineRequest) -> GeneratedAnswerResponse:
    """Run retrieval and generate a grounded philosophical response."""
    try:
        cache_manager = getattr(request.app.state, "cache_manager", None)
        engine = get_adaptive_engine(cache_manager=cache_manager)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize adaptive engine: {exc}",
        ) from exc

    try:
        generated = engine.answer(payload.query)
        return GeneratedAnswerResponse(
            original_query=generated.original_query,
            route=generated.route,
            answer=generated.answer,
            cited_verses=generated.cited_verses,
            warnings=generated.warnings,
            contexts=[
                {
                    "chapter": context.chapter,
                    "verse": context.verse,
                    "shloka": context.shloka,
                    "translation": context.translation,
                    "interpretation": context.interpretation,
                    "topics": context.topics,
                }
                for context in generated.contexts
            ],
            used_rag=generated.used_rag,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Answer generation failed: {exc}",
        ) from exc
