"""Core cache management with Redis backend and in-memory fallback."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CacheEntry(BaseModel):
    """Typed cache entry with metadata."""

    value: Any = Field(..., description="Cached value (JSON-serializable)")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = Field(None, description="TTL expiration time")
    hit_count: int = Field(0, ge=0)
    access_count: int = Field(0, ge=0)

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def record_hit(self) -> None:
        """Record a cache hit."""
        self.hit_count += 1
        self.access_count += 1

    def record_miss(self) -> None:
        """Record a cache miss."""
        self.access_count += 1


class CacheManager:
    """
    Main cache orchestrator.
    
    Features:
    - In-memory fallback (always available)
    - Optional Redis backend (if configured)
    - TTL support
    - Hit tracking
    - Size limits
    - Graceful degradation
    """

    def __init__(self, max_memory_items: int = 1000, default_ttl_seconds: int = 3600) -> None:
        """
        Initialize cache manager.
        
        Args:
            max_memory_items: Max items in in-memory cache (LRU eviction)
            default_ttl_seconds: Default TTL for entries
        """
        self.max_memory_items = max_memory_items
        self.default_ttl_seconds = default_ttl_seconds
        
        # In-memory cache (always available)
        self._memory_cache: dict[str, CacheEntry] = {}
        
        # Redis client (optional, lazy-loaded)
        self._redis_client: Optional[Any] = None
        self._redis_enabled = os.getenv("REDIS_URL", "").strip() != ""
        
        logger.info(
            "[CACHE] Initialized CacheManager (max_memory=%d, default_ttl=%ds, redis=%s)",
            max_memory_items,
            default_ttl_seconds,
            "enabled" if self._redis_enabled else "disabled",
        )

    def _ensure_redis(self) -> Optional[Any]:
        """Lazy-load Redis client if enabled."""
        if not self._redis_enabled or self._redis_client is not None:
            return self._redis_client
        
        try:
            import redis
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            self._redis_client = redis.from_url(redis_url, decode_responses=True)
            self._redis_client.ping()
            logger.info("[CACHE] Redis connected successfully")
            return self._redis_client
        except Exception as exc:
            logger.warning("[CACHE] Redis connection failed, using memory-only cache: %s", exc)
            self._redis_enabled = False
            return None

    def get(
        self,
        key: str,
        value_type: type | None = None,
    ) -> Optional[Any]:
        """
        Retrieve value from cache.
        
        Args:
            key: Cache key
            value_type: Expected type for validation
            
        Returns:
            Cached value or None if not found/expired
        """
        # Try Redis first
        if self._redis_enabled:
            try:
                redis_client = self._ensure_redis()
                if redis_client:
                    raw = redis_client.get(key)
                    if raw:
                        value = json.loads(raw)
                        logger.debug("[CACHE] Hit (Redis): %s", key)
                        return value
            except Exception as exc:
                logger.warning("[CACHE] Redis GET failed: %s", exc)

        # Fall back to memory
        entry = self._memory_cache.get(key)
        if entry is None:
            logger.debug("[CACHE] Miss: %s", key)
            return None

        if entry.is_expired():
            logger.debug("[CACHE] Expired: %s", key)
            del self._memory_cache[key]
            return None

        entry.record_hit()
        logger.debug("[CACHE] Hit (Memory): %s (count=%d)", key, entry.hit_count)
        return entry.value

    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """
        Store value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: TTL override (None = use default)
        """
        ttl = ttl_seconds or self.default_ttl_seconds
        expires_at = datetime.utcnow() + timedelta(seconds=ttl) if ttl > 0 else None

        entry = CacheEntry(
            value=value,
            expires_at=expires_at,
        )

        # Store in Redis if enabled
        if self._redis_enabled:
            try:
                redis_client = self._ensure_redis()
                if redis_client:
                    json_val = json.dumps(value)
                    redis_client.setex(
                        key,
                        ttl if ttl > 0 else 86400,  # 24h max
                        json_val,
                    )
                    logger.debug("[CACHE] Set (Redis): %s (ttl=%ds)", key, ttl)
                    return
            except Exception as exc:
                logger.warning("[CACHE] Redis SET failed, using memory: %s", exc)

        # Store in memory
        if len(self._memory_cache) >= self.max_memory_items:
            self._evict_lru()

        self._memory_cache[key] = entry
        logger.debug("[CACHE] Set (Memory): %s (ttl=%ds)", key, ttl)

    def delete(self, key: str) -> None:
        """Delete entry from cache."""
        # Delete from Redis if enabled
        if self._redis_enabled:
            try:
                redis_client = self._ensure_redis()
                if redis_client:
                    redis_client.delete(key)
                    logger.debug("[CACHE] Deleted (Redis): %s", key)
            except Exception as exc:
                logger.warning("[CACHE] Redis DELETE failed: %s", exc)

        # Delete from memory
        if key in self._memory_cache:
            del self._memory_cache[key]
            logger.debug("[CACHE] Deleted (Memory): %s", key)

    def clear(self) -> None:
        """Clear entire cache."""
        self._memory_cache.clear()
        logger.info("[CACHE] Memory cache cleared")

        if self._redis_enabled:
            try:
                redis_client = self._ensure_redis()
                if redis_client:
                    redis_client.flushdb()
                    logger.info("[CACHE] Redis cache cleared")
            except Exception as exc:
                logger.warning("[CACHE] Redis FLUSHDB failed: %s", exc)

    def _evict_lru(self) -> None:
        """Evict least-recently-used item from memory cache."""
        if not self._memory_cache:
            return

        lru_key = min(
            self._memory_cache.keys(),
            key=lambda k: (self._memory_cache[k].access_count, self._memory_cache[k].created_at),
        )
        del self._memory_cache[lru_key]
        logger.debug("[CACHE] LRU eviction: %s", lru_key)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_hits = sum(e.hit_count for e in self._memory_cache.values())
        total_accesses = sum(e.access_count for e in self._memory_cache.values())
        hit_rate = total_hits / total_accesses if total_accesses > 0 else 0.0

        return {
            "memory_items": len(self._memory_cache),
            "memory_max": self.max_memory_items,
            "total_hits": total_hits,
            "total_accesses": total_accesses,
            "hit_rate": hit_rate,
            "redis_enabled": self._redis_enabled,
        }
