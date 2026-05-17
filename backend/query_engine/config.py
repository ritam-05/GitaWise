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
    qdrant_api_key: str = Field(default_factory=lambda: os.getenv("QDRANT_API_KEY", ""))
    qdrant_endpoint: str = Field(default_factory=lambda: os.getenv("QDRANT_ENDPOINT", ""))
    groq_model_name: str = "llama-3.3-70b-versatile"
    # OPTIMIZED: Smaller model for extraction/classification tasks (analyzer, router)
    # This reduces token usage and cost while maintaining quality for structured extraction
    groq_analyzer_model_name: str = "llama-3.1-8b-instant"
    embedding_model_name: str = "BAAI/bge-large-en-v1.5"
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"
    qdrant_collection_name: str = "gita_verses"
    qdrant_top_k_per_problem: int = 5
    final_top_k: int = 6
    retrieval_confidence_threshold: float = 0.65
    qdrant_timeout_seconds: int = 60
    groq_temperature: float = 0.0
    groq_max_retries: int = 2
    reranker_batch_size: int = 16
    # When performing hybrid reranking we first filter by embedding similarity
    # using the retriever's scores and then LLM-rerank the top N candidates.
    reranker_embed_filter_top_k: int = 20
    # Number of top contexts (chunks) to include in the final generation prompt.
    generation_context_top_k: int = 3
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
