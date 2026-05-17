"""Intelligent cache key generation and hashing for semantic matching."""

from __future__ import annotations

import hashlib
from typing import Any


class CacheKeys:
    """Generate deterministic, semantic-aware cache keys."""

    # Prefix constants
    PREFIX_ROUTE = "route"
    PREFIX_DECOMPOSITION = "decomp"
    PREFIX_EMOTION = "emotion"
    PREFIX_RETRIEVAL_QUERY = "ret_query"
    PREFIX_RETRIEVAL = "retrieval"
    PREFIX_RERANK = "rerank"
    PREFIX_EMBEDDING = "embed"
    PREFIX_SESSION_CONTEXT = "session_ctx"
    PREFIX_CONVERSATION = "convo"

    @staticmethod
    def hash_query(query: str) -> str:
        """Generate deterministic hash for a query string.
        
        Args:
            query: User query text
            
        Returns:
            SHA-256 hash (first 12 chars for readability)
        """
        normalized = query.strip().lower()
        hash_obj = hashlib.sha256(normalized.encode("utf-8"))
        return hash_obj.hexdigest()[:12]

    @staticmethod
    def route_key(query: str) -> str:
        """Cache key for route decision."""
        query_hash = CacheKeys.hash_query(query)
        return f"{CacheKeys.PREFIX_ROUTE}:{query_hash}"

    @staticmethod
    def decomposition_key(query: str) -> str:
        """Cache key for query decomposition result."""
        query_hash = CacheKeys.hash_query(query)
        return f"{CacheKeys.PREFIX_DECOMPOSITION}:{query_hash}"

    @staticmethod
    def emotion_key(query: str) -> str:
        """Cache key for emotion analysis result."""
        query_hash = CacheKeys.hash_query(query)
        return f"{CacheKeys.PREFIX_EMOTION}:{query_hash}"

    @staticmethod
    def retrieval_query_key(problem: str, emotion: str) -> str:
        """Cache key for generated retrieval query."""
        combined = f"{problem}|{emotion}".strip().lower()
        hash_val = hashlib.sha256(combined.encode("utf-8")).hexdigest()[:8]
        return f"{CacheKeys.PREFIX_RETRIEVAL_QUERY}:{hash_val}"

    @staticmethod
    def retrieval_key(retrieval_query: str) -> str:
        """Cache key for retrieval results (embedding-based)."""
        query_hash = CacheKeys.hash_query(retrieval_query)
        return f"{CacheKeys.PREFIX_RETRIEVAL}:{query_hash}"

    @staticmethod
    def rerank_key(retrieval_results_hash: str, original_query: str) -> str:
        """Cache key for reranking results."""
        query_hash = CacheKeys.hash_query(original_query)
        return f"{CacheKeys.PREFIX_RERANK}:{retrieval_results_hash}:{query_hash}"

    @staticmethod
    def embedding_key(text: str) -> str:
        """Cache key for embedding vector."""
        text_hash = CacheKeys.hash_query(text)
        return f"{CacheKeys.PREFIX_EMBEDDING}:{text_hash}"

    @staticmethod
    def session_context_key(session_id: str) -> str:
        """Cache key for session-level context and memory."""
        return f"{CacheKeys.PREFIX_SESSION_CONTEXT}:{session_id}"

    @staticmethod
    def conversation_key(session_id: str, turn_num: int) -> str:
        """Cache key for conversation turn history."""
        return f"{CacheKeys.PREFIX_CONVERSATION}:{session_id}:turn_{turn_num}"

    @staticmethod
    def global_model_key(model_type: str) -> str:
        """Cache key for globally-loaded models (singleton)."""
        return f"global:model:{model_type}"

    @staticmethod
    def similarity_query_hash(query: str, hash_length: int = 6) -> str:
        """Shorter hash for semantic similarity matching.
        
        Useful for finding "similar enough" queries in cache.
        
        Args:
            query: Query text
            hash_length: Length of returned hash
            
        Returns:
            Short hash for prefix-based similarity matching
        """
        normalized = query.strip().lower()
        hash_obj = hashlib.sha256(normalized.encode("utf-8"))
        return hash_obj.hexdigest()[:hash_length]
