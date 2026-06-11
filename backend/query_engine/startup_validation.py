"""Startup validation for embedding and Qdrant compatibility."""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient

from .config import QueryEngineConfig, get_logger
from .embedding_model import (
    get_embedding_dimension,
    get_embedding_model,
    initialize_embedding_model,
)
from .reranker_model import initialize_reranker_model

LOGGER = get_logger(__name__)


def _extract_collection_dimension(collection_info: Any) -> int:
    """Return the dense-vector dimension from a Qdrant collection description."""
    vectors_config = collection_info.config.params.vectors

    if hasattr(vectors_config, "size"):
        return int(vectors_config.size)

    if isinstance(vectors_config, dict):
        if not vectors_config:
            raise RuntimeError("Qdrant collection has no vector configuration.")
        first_vector = next(iter(vectors_config.values()))
        if hasattr(first_vector, "size"):
            return int(first_vector.size)

    raise RuntimeError(
        "Could not determine Qdrant collection vector dimension from collection info."
    )


def _validate_reranker_config(config: QueryEngineConfig) -> None:
    """Validate reranker settings that affect retrieval/reranking flow."""
    if not config.reranker_model_name.strip():
        raise RuntimeError("Reranker model name is empty.")
    if config.reranker_batch_size <= 0:
        raise RuntimeError("Reranker batch size must be greater than 0.")
    if config.reranker_embed_filter_top_k < config.final_top_k:
        raise RuntimeError(
            "reranker_embed_filter_top_k must be greater than or equal to final_top_k."
        )


def validate_query_engine_startup(config: QueryEngineConfig) -> None:
    """Fail fast if runtime embeddings and the Qdrant collection are incompatible."""
    initialize_embedding_model(config.embedding_model_name)
    embedding_model = get_embedding_model()
    embedding_dimension = get_embedding_dimension(embedding_model)

    qdrant_client = QdrantClient(
        url=config.qdrant_endpoint,
        api_key=config.qdrant_api_key,
        timeout=config.qdrant_timeout_seconds,
    )
    collection_info = qdrant_client.get_collection(config.qdrant_collection_name)
    collection_dimension = _extract_collection_dimension(collection_info)

    LOGGER.info(
        "[STARTUP_VALIDATION] Embedding model=%s dimension=%s; "
        "Qdrant collection=%s dimension=%s",
        config.embedding_model_name,
        embedding_dimension,
        config.qdrant_collection_name,
        collection_dimension,
    )

    if collection_dimension != embedding_dimension:
        raise RuntimeError(
            f"Embedding model dimension ({embedding_dimension}) "
            f"does not match Qdrant collection dimension ({collection_dimension}). "
            "Rebuild the collection or switch embedding models."
        )

    _validate_reranker_config(config)
    LOGGER.info(
        "[STARTUP_VALIDATION] Reranker model=%s batch_size=%s embed_filter_top_k=%s",
        config.reranker_model_name,
        config.reranker_batch_size,
        config.reranker_embed_filter_top_k,
    )

    # Warm up reranker at startup so the first request does not trigger
    # a slow lazy-load that times out the Next.js proxy (ECONNRESET).
    try:
        LOGGER.info("[STARTUP_VALIDATION] Pre-loading reranker model: %s", config.reranker_model_name)
        initialize_reranker_model(config.reranker_model_name)
        LOGGER.info("[STARTUP_VALIDATION] ✓ Reranker model pre-loaded successfully")
    except Exception as exc:
        # Log but don't crash startup — reranking will fail gracefully at request time
        LOGGER.warning("[STARTUP_VALIDATION] Reranker pre-load failed (will retry on first request): %s", exc)
