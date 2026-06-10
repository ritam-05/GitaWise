"""Core cache management with Supabase persistence and in-memory fallback."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Optional

import requests
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
    - Optional Supabase persistence backend (if configured)
    - TTL support
    - Hit tracking
    - Size limits
    - Graceful degradation
    """

    def __init__(
        self,
        max_memory_items: int = 1000,
        default_ttl_seconds: int = 3600,
        supabase_url: str | None = None,
        supabase_key: str | None = None,
        supabase_table: str | None = None,
    ) -> None:
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
        
        # Supabase persistence (optional, lazy-loaded)
        self._supabase_url = (supabase_url or os.getenv("SUPABASE_URL", "")).strip().rstrip("/")
        self._supabase_key = (
            (supabase_key or "").strip()
            or os.getenv("SUPABASE_SERVICE_KEY", "").strip()
            or os.getenv("SUPABASE_ANON_KEY", "").strip()
        )
        self._supabase_table = (
            (supabase_table or os.getenv("SUPABASE_CACHE_TABLE", "cache_entries")).strip()
            or "cache_entries"
        )
        self._supabase_enabled = bool(self._supabase_url and self._supabase_key)
        self._supabase_ready = False
        self._supabase_timeout_seconds = int(os.getenv("SUPABASE_TIMEOUT_SECONDS", "10"))
        self._supabase_retry_seconds = int(os.getenv("SUPABASE_RETRY_SECONDS", "60"))
        self._last_supabase_failure: datetime | None = None
        
        logger.info(
            "[CACHE] Initialized CacheManager (max_memory=%d, default_ttl=%ds, supabase=%s, table=%s)",
            max_memory_items,
            default_ttl_seconds,
            "enabled" if self._supabase_enabled else "disabled",
            self._supabase_table,
        )

    def _supabase_headers(self, prefer: str | None = None) -> dict[str, str]:
        headers = {
            "apikey": self._supabase_key,
            "Authorization": f"Bearer {self._supabase_key}",
            "Content-Type": "application/json",
        }
        if prefer:
            headers["Prefer"] = prefer
        return headers

    def _supabase_endpoint(self) -> str:
        return f"{self._supabase_url}/rest/v1/{self._supabase_table}"

    def _ensure_supabase(self) -> bool:
        """Lazy-verify that Supabase is reachable for cache persistence."""
        if not self._supabase_enabled:
            return False
        if self._supabase_ready:
            return True
        if (
            self._last_supabase_failure
            and (datetime.utcnow() - self._last_supabase_failure).total_seconds()
            < self._supabase_retry_seconds
        ):
            return False

        try:
            response = requests.get(
                self._supabase_endpoint(),
                headers=self._supabase_headers(),
                params={"select": "key", "limit": 1},
                timeout=self._supabase_timeout_seconds,
            )
            response.raise_for_status()
            self._supabase_ready = True
            self._last_supabase_failure = None
            logger.info("[CACHE] Supabase cache connected successfully")
            return True
        except Exception as exc:
            logger.warning(
                "[CACHE] Supabase cache unavailable, using memory-only cache until retry: %s",
                exc,
            )
            self._supabase_ready = False
            self._last_supabase_failure = datetime.utcnow()
            return False

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        """Ensure values are JSON-safe for persistence."""
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, dict):
            return {str(k): CacheManager._serialize_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [CacheManager._serialize_value(item) for item in value]
        return value

    @staticmethod
    def _parse_datetime(value: Any) -> Optional[datetime]:
        if not value:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            normalized = value.replace("Z", "+00:00")
            return datetime.fromisoformat(normalized)
        return None

    def _supabase_get(self, key: str) -> Optional[Any]:
        if not self._ensure_supabase():
            return None

        now_iso = datetime.utcnow().isoformat()
        response = requests.get(
            self._supabase_endpoint(),
            headers=self._supabase_headers(),
            params={
                "select": "value,expires_at",
                "key": f"eq.{key}",
                "or": f"(expires_at.is.null,expires_at.gt.{now_iso})",
                "limit": 1,
            },
            timeout=self._supabase_timeout_seconds,
        )
        response.raise_for_status()
        rows = response.json()
        if not rows:
            return None
        return rows[0].get("value")

    def _supabase_set(self, key: str, value: Any, ttl_seconds: int) -> bool:
        if not self._ensure_supabase():
            return False

        expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds) if ttl_seconds > 0 else None
        payload = {
            "key": key,
            "value": self._serialize_value(value),
            "expires_at": expires_at.isoformat() if expires_at else None,
        }
        response = requests.post(
            self._supabase_endpoint(),
            headers=self._supabase_headers(prefer="resolution=merge-duplicates,return=minimal"),
            params={"on_conflict": "key"},
            data=json.dumps(payload),
            timeout=self._supabase_timeout_seconds,
        )
        response.raise_for_status()
        return True

    def _supabase_delete(self, key: str) -> None:
        if not self._ensure_supabase():
            return
        response = requests.delete(
            self._supabase_endpoint(),
            headers=self._supabase_headers(),
            params={"key": f"eq.{key}"},
            timeout=self._supabase_timeout_seconds,
        )
        response.raise_for_status()

    def _supabase_clear(self) -> None:
        if not self._ensure_supabase():
            return
        response = requests.delete(
            self._supabase_endpoint(),
            headers=self._supabase_headers(),
            params={"key": "neq.__never__"},
            timeout=self._supabase_timeout_seconds,
        )
        response.raise_for_status()

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
        # Try Supabase first
        if self._supabase_enabled:
            try:
                value = self._supabase_get(key)
                if value is not None:
                    logger.debug("[CACHE] Hit (Supabase): %s", key)
                    return value
            except Exception as exc:
                logger.warning("[CACHE] Supabase GET failed: %s", exc)

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

        # Store in Supabase if enabled
        if self._supabase_enabled:
            try:
                if self._supabase_set(key, value, ttl):
                    logger.debug("[CACHE] Set (Supabase): %s (ttl=%ds)", key, ttl)
                    return
            except Exception as exc:
                logger.warning("[CACHE] Supabase SET failed, using memory: %s", exc)

        # Store in memory
        if len(self._memory_cache) >= self.max_memory_items:
            self._evict_lru()

        self._memory_cache[key] = entry
        logger.debug("[CACHE] Set (Memory): %s (ttl=%ds)", key, ttl)

    def delete(self, key: str) -> None:
        """Delete entry from cache."""
        # Delete from Supabase if enabled
        if self._supabase_enabled:
            try:
                self._supabase_delete(key)
                logger.debug("[CACHE] Deleted (Supabase): %s", key)
            except Exception as exc:
                logger.warning("[CACHE] Supabase DELETE failed: %s", exc)

        # Delete from memory
        if key in self._memory_cache:
            del self._memory_cache[key]
            logger.debug("[CACHE] Deleted (Memory): %s", key)

    def clear(self) -> None:
        """Clear entire cache."""
        self._memory_cache.clear()
        logger.info("[CACHE] Memory cache cleared")

        if self._supabase_enabled:
            try:
                self._supabase_clear()
                logger.info("[CACHE] Supabase cache cleared")
            except Exception as exc:
                logger.warning("[CACHE] Supabase cache clear failed: %s", exc)

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
            "supabase_enabled": self._supabase_enabled,
            "supabase_table": self._supabase_table,
        }
