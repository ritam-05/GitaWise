"""Semantic caching for intelligent similarity-based cache matching."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from .cache_keys import CacheKeys
from .cache_manager import CacheManager

logger = logging.getLogger(__name__)


class SemanticCache:
    """
    Semantic-aware cache matching.
    
    Matches queries based on semantic similarity rather than exact string matching.
    Useful for matching "I fear failure" with "I am afraid of failing".
    """

    def __init__(
        self,
        cache_manager: CacheManager,
        similarity_threshold: float = 0.85,
    ) -> None:
        """
        Initialize semantic cache.
        
        Args:
            cache_manager: Global cache manager
            similarity_threshold: Cosine similarity threshold for matching (0-1)
        """
        self.cache_manager = cache_manager
        self.similarity_threshold = similarity_threshold
        
        # In-memory index of recent cache entries for semantic search
        self._semantic_index: dict[str, tuple[str, np.ndarray]] = {}
        
        logger.info(
            "[SEMANTIC_CACHE] Initialized (threshold=%.2f)",
            similarity_threshold,
        )

    def cache_embedding(
        self,
        text: str,
        embedding: list[float] | np.ndarray,
        value: Any,
    ) -> str:
        """
        Cache value with embedding for semantic lookup.
        
        Args:
            text: Original text
            embedding: Embedding vector
            value: Value to cache
            
        Returns:
            Cache key used
        """
        cache_key = CacheKeys.embedding_key(text)
        
        # Store embedding and value
        embedding_array = np.array(embedding) if not isinstance(embedding, np.ndarray) else embedding
        self._semantic_index[cache_key] = (text, embedding_array)
        
        # Store in main cache
        self.cache_manager.set(
            cache_key,
            {
                "text": text,
                "embedding": embedding_array.tolist(),
                "value": value,
            },
        )
        
        logger.debug("[SEMANTIC_CACHE] Cached embedding for: %s", text[:50])
        return cache_key

    def find_similar(
        self,
        query: str,
        query_embedding: list[float] | np.ndarray,
    ) -> Optional[tuple[str, Any, float]]:
        """
        Find semantically similar cached value.
        
        Args:
            query: Query text
            query_embedding: Query embedding vector
            
        Returns:
            Tuple of (original_text, cached_value, similarity_score) or None
        """
        if not self._semantic_index:
            return None

        query_vector = np.array(query_embedding)
        
        best_match = None
        best_similarity = self.similarity_threshold

        # Find best match in semantic index
        for cache_key, (cached_text, cached_embedding) in self._semantic_index.items():
            try:
                # Compute cosine similarity
                similarity = self._cosine_similarity(query_vector, cached_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = (cache_key, cached_text, similarity)
            except Exception as exc:
                logger.warning("[SEMANTIC_CACHE] Similarity computation failed: %s", exc)
                continue

        if best_match:
            cache_key, cached_text, similarity = best_match
            
            # Retrieve cached value
            cached_data = self.cache_manager.get(cache_key)
            if cached_data:
                logger.info(
                    "[SEMANTIC_CACHE] Similar match found: %.3f (cached: '%s', query: '%s')",
                    similarity,
                    cached_text[:40],
                    query[:40],
                )
                return (cached_text, cached_data.get("value"), similarity)

        logger.debug("[SEMANTIC_CACHE] No similar matches found (threshold=%.2f)", self.similarity_threshold)
        return None

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
        except Exception:
            return 0.0

    def clear_index(self) -> None:
        """Clear semantic index (but keep main cache)."""
        self._semantic_index.clear()
        logger.debug("[SEMANTIC_CACHE] Semantic index cleared")

    def get_index_stats(self) -> dict[str, Any]:
        """Get semantic index statistics."""
        return {
            "indexed_items": len(self._semantic_index),
            "similarity_threshold": self.similarity_threshold,
        }
