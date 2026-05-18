"""Configuration for the GitaWise query engine."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


class QueryEngineConfig(BaseModel):
    """Runtime configuration for decomposition, retrieval, and reranking."""

    groq_api_key: str = Field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    sarvam_api_key: str = Field(default_factory=lambda: os.getenv("SARVAM_API_KEY", ""))
    qdrant_api_key: str = Field(default_factory=lambda: os.getenv("QDRANT_API_KEY", ""))
    qdrant_endpoint: str = Field(default_factory=lambda: os.getenv("QDRANT_ENDPOINT", ""))
    supabase_url: str = Field(default_factory=lambda: os.getenv("SUPABASE_URL", ""))
    supabase_anon_key: str = Field(default_factory=lambda: os.getenv("SUPABASE_ANON_KEY", ""))
    supabase_service_key: str = Field(default_factory=lambda: os.getenv("SUPABASE_SERVICE_KEY", ""))
    supabase_cache_table: str = Field(
        default_factory=lambda: os.getenv("SUPABASE_CACHE_TABLE", "cache_entries")
    )
    groq_model_name: str = "llama-3.3-70b-versatile"
    # OPTIMIZED: Smaller model for extraction/classification tasks (analyzer, router)
    # This reduces token usage and cost while maintaining quality for structured extraction
    groq_analyzer_model_name: str = "llama-3.1-8b-instant"
    sarvam_model_name: str = "sarvam-m"
    sarvam_fallback_model_name: str = "sarvam-m"
    embedding_model_name: str = "BAAI/bge-large-en-v1.5"
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"
    qdrant_collection_name: str = "gita_verses"
    qdrant_top_k_per_problem: int = 5
    final_top_k: int = 6
    retrieval_confidence_threshold: float = 0.65
    qdrant_timeout_seconds: int = 60
    groq_temperature: float = 0.0
    groq_max_retries: int = 2
    sarvam_temperature: float = 0.2
    sarvam_max_retries: int = 2
    sarvam_max_tokens: int = 1024
    sarvam_timeout_seconds: int = 60
    reranker_batch_size: int = 16
    # When performing hybrid reranking we first filter by embedding similarity
    # using the retriever's scores and then LLM-rerank the top N candidates.
    reranker_embed_filter_top_k: int = 20
    # Number of top contexts (chunks) to include in the final generation prompt.
    generation_context_top_k: int = 3
    
    # Session configuration (conversation chaining)
    session_ttl_seconds: int = Field(
        default_factory=lambda: int(os.getenv("SESSION_TTL_SECONDS", "3600")),
        description="Session time-to-live in seconds (default: 1 hour)"
    )
    session_max_stored_turns: int = Field(
        default_factory=lambda: int(os.getenv("SESSION_MAX_STORED_TURNS", "10")),
        description="Maximum conversation turns to store in session memory"
    )
    session_context_turns: int = Field(
        default_factory=lambda: int(os.getenv("SESSION_CONTEXT_TURNS", "5")),
        description="Number of recent turns to include in LLM context"
    )
    session_cleanup_interval_seconds: int = Field(
        default_factory=lambda: int(os.getenv("SESSION_CLEANUP_INTERVAL_SECONDS", "1800")),
        description="How often to cleanup expired sessions (default: 30 minutes)"
    )
    
    log_level: str = "INFO"

    @field_validator("groq_api_key", "qdrant_api_key", "qdrant_endpoint")
    @classmethod
    def _validate_required_env(cls, value: str, info: object) -> str:
        if not value.strip():
            field_name = getattr(info, "field_name", "configuration")
            raise ValueError(f"Missing required configuration value for '{field_name}'.")
        return value

    @field_validator("qdrant_top_k_per_problem")
    @classmethod
    def _validate_top_k(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("qdrant_top_k_per_problem must be greater than 0.")
        return value

    @field_validator("final_top_k")
    @classmethod
    def _validate_final_top_k(cls, value: int) -> int:
        if not 1 <= value <= 8:
            raise ValueError("final_top_k must be between 1 and 8.")
        return value

    @field_validator("reranker_embed_filter_top_k")
    @classmethod
    def _validate_reranker_embed_filter_top_k(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("reranker_embed_filter_top_k must be greater than 0.")
        return value

    @field_validator("generation_context_top_k")
    @classmethod
    def _validate_generation_context_top_k(cls, value: int) -> int:
        if not 1 <= value <= 8:
            raise ValueError("generation_context_top_k must be between 1 and 8.")
        return value

    @field_validator("retrieval_confidence_threshold")
    @classmethod
    def _validate_threshold(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("retrieval_confidence_threshold must be between 0.0 and 1.0.")
        return value

    @field_validator("sarvam_temperature")
    @classmethod
    def _validate_sarvam_temperature(cls, value: float) -> float:
        if not 0.0 <= value <= 2.0:
            raise ValueError("sarvam_temperature must be between 0.0 and 2.0.")
        return value

    @field_validator("sarvam_max_retries", "sarvam_max_tokens", "sarvam_timeout_seconds")
    @classmethod
    def _validate_positive_ints(cls, value: int, info: object) -> int:
        if value <= 0:
            field_name = getattr(info, "field_name", "configuration")
            raise ValueError(f"{field_name} must be greater than 0.")
        return value

    @field_validator("session_ttl_seconds")
    @classmethod
    def _validate_session_ttl(cls, value: int) -> int:
        if value < 60:
            raise ValueError("session_ttl_seconds must be at least 60 seconds (1 minute).")
        if value > 86400:
            raise ValueError("session_ttl_seconds must not exceed 86400 seconds (24 hours).")
        return value

    @field_validator("session_max_stored_turns")
    @classmethod
    def _validate_session_max_turns(cls, value: int) -> int:
        if not 5 <= value <= 100:
            raise ValueError("session_max_stored_turns must be between 5 and 100.")
        return value

    @field_validator("session_context_turns")
    @classmethod
    def _validate_session_context_turns(cls, value: int) -> int:
        if not 1 <= value <= 20:
            raise ValueError("session_context_turns must be between 1 and 20.")
        return value

    @field_validator("session_cleanup_interval_seconds")
    @classmethod
    def _validate_session_cleanup_interval(cls, value: int) -> int:
        if value < 60:
            raise ValueError("session_cleanup_interval_seconds must be at least 60 seconds.")
        if value > 86400:
            raise ValueError("session_cleanup_interval_seconds must not exceed 86400 seconds.")
        return value


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Create a consistent logger for query-engine modules."""
    logger = logging.getLogger(name)
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def load_query_engine_config() -> QueryEngineConfig:
    """Load and validate runtime settings."""
    return QueryEngineConfig()
