"""Session-aware cache layer for conversational continuity and memory."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Optional

from pydantic import BaseModel, Field

from .cache_keys import CacheKeys
from .cache_manager import CacheManager

logger = logging.getLogger(__name__)


class ConversationTurn(BaseModel):
    """Single conversation turn."""

    turn_num: int
    user_query: str
    route: str
    problems: list[dict[str, str]] = Field(default_factory=list)
    emotions: list[str] = Field(default_factory=list)
    retrieved_verses: list[dict[str, Any]] = Field(default_factory=list)
    response: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SessionMemory(BaseModel):
    """Lightweight session memory for conversational continuity."""

    session_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    
    # Recent conversation context
    recent_turns: list[ConversationTurn] = Field(default_factory=list, max_length=10)
    
    # Themes and patterns
    detected_themes: list[str] = Field(default_factory=list)
    detected_emotions: list[str] = Field(default_factory=list)
    dominant_route: Optional[str] = None
    
    # Statistics
    total_turns: int = 0
    total_verses_retrieved: int = 0

    def add_turn(self, turn: ConversationTurn) -> None:
        """Add conversation turn to memory."""
        self.recent_turns.append(turn)
        if len(self.recent_turns) > 10:
            self.recent_turns.pop(0)
        self.total_turns += 1
        self.last_accessed = datetime.utcnow()

    def update_themes(self, themes: list[str], emotions: list[str], route: str) -> None:
        """Update detected patterns."""
        self.detected_themes = list(set(self.detected_themes + themes))[:5]
        self.detected_emotions = list(set(self.detected_emotions + emotions))[:8]
        self.dominant_route = route
        self.last_accessed = datetime.utcnow()

    def is_expired(self, ttl_seconds: int = 3600) -> bool:
        """Check if session has expired."""
        expiration = self.last_accessed + timedelta(seconds=ttl_seconds)
        return datetime.utcnow() > expiration


class SessionCache:
    """
    Per-session cache layer for intelligent conversational continuity.
    
    Maintains:
    - Conversation history
    - Detected themes and emotions
    - Reusable decompositions
    - Retrieved verse sets
    - Session-level context
    """

    def __init__(self, cache_manager: CacheManager, session_ttl_seconds: int = 3600) -> None:
        """
        Initialize session cache.
        
        Args:
            cache_manager: Global cache manager instance
            session_ttl_seconds: Session TTL before expiration
        """
        self.cache_manager = cache_manager
        self.session_ttl_seconds = session_ttl_seconds
        logger.info("[SESSION_CACHE] Initialized (ttl=%ds)", session_ttl_seconds)

    def get_or_create_session(self, session_id: str) -> SessionMemory:
        """Get or create session memory."""
        cache_key = CacheKeys.session_context_key(session_id)
        
        # Try to retrieve existing session
        cached = self.cache_manager.get(cache_key)
        if cached:
            session = SessionMemory(**cached) if isinstance(cached, dict) else cached
            logger.info("[SESSION] Retrieved session: %s", session_id)
            return session

        # Create new session
        session = SessionMemory(session_id=session_id)
        self.save_session(session)
        logger.info("[SESSION] Created new session: %s", session_id)
        return session

    def save_session(self, session: SessionMemory) -> None:
        """Save session to cache."""
        cache_key = CacheKeys.session_context_key(session.session_id)
        self.cache_manager.set(
            cache_key,
            session.model_dump(),
            ttl_seconds=self.session_ttl_seconds,
        )
        logger.debug("[SESSION] Saved session: %s", session.session_id)

    def cache_decomposition(
        self,
        session_id: str,
        query: str,
        problems: list[dict[str, str]],
    ) -> None:
        """Cache query decomposition within session context."""
        cache_key = CacheKeys.decomposition_key(query)
        self.cache_manager.set(
            cache_key,
            {
                "query": query,
                "problems": problems,
                "session_id": session_id,
            },
            ttl_seconds=self.session_ttl_seconds,
        )
        logger.debug("[SESSION] Cached decomposition for query: %s", query[:50])

    def get_cached_decomposition(self, query: str) -> Optional[list[dict[str, str]]]:
        """Retrieve cached decomposition."""
        cache_key = CacheKeys.decomposition_key(query)
        cached = self.cache_manager.get(cache_key)
        if cached and isinstance(cached, dict):
            logger.debug("[SESSION] Hit: decomposition for query: %s", query[:50])
            return cached.get("problems")
        return None

    def cache_emotions(
        self,
        session_id: str,
        query: str,
        emotions: list[str],
    ) -> None:
        """Cache emotion analysis within session."""
        cache_key = CacheKeys.emotion_key(query)
        self.cache_manager.set(
            cache_key,
            {
                "query": query,
                "emotions": emotions,
                "session_id": session_id,
            },
            ttl_seconds=self.session_ttl_seconds,
        )
        logger.debug("[SESSION] Cached emotions for query: %s", query[:50])

    def get_cached_emotions(self, query: str) -> Optional[list[str]]:
        """Retrieve cached emotions."""
        cache_key = CacheKeys.emotion_key(query)
        cached = self.cache_manager.get(cache_key)
        if cached and isinstance(cached, dict):
            logger.debug("[SESSION] Hit: emotions for query: %s", query[:50])
            return cached.get("emotions")
        return None

    def cache_retrieval(
        self,
        session_id: str,
        retrieval_query: str,
        verses: list[dict[str, Any]],
    ) -> None:
        """Cache retrieval results."""
        cache_key = CacheKeys.retrieval_key(retrieval_query)
        self.cache_manager.set(
            cache_key,
            {
                "retrieval_query": retrieval_query,
                "verses": verses,
                "session_id": session_id,
                "count": len(verses),
            },
            ttl_seconds=self.session_ttl_seconds,
        )
        logger.debug("[SESSION] Cached %d retrieved verses for query: %s", len(verses), retrieval_query[:50])

    def get_cached_retrieval(self, retrieval_query: str) -> Optional[list[dict[str, Any]]]:
        """Retrieve cached verses."""
        cache_key = CacheKeys.retrieval_key(retrieval_query)
        cached = self.cache_manager.get(cache_key)
        if cached and isinstance(cached, dict):
            verses = cached.get("verses", [])
            logger.debug("[SESSION] Hit: %d cached verses for query: %s", len(verses), retrieval_query[:50])
            return verses
        return None

    def cache_reranked_verses(
        self,
        session_id: str,
        retrieval_hash: str,
        original_query: str,
        reranked_verses: list[dict[str, Any]],
    ) -> None:
        """Cache reranked results."""
        cache_key = CacheKeys.rerank_key(retrieval_hash, original_query)
        self.cache_manager.set(
            cache_key,
            {
                "retrieval_hash": retrieval_hash,
                "original_query": original_query,
                "verses": reranked_verses,
                "session_id": session_id,
            },
            ttl_seconds=self.session_ttl_seconds,
        )
        logger.debug("[SESSION] Cached reranked results (%d verses)", len(reranked_verses))

    def get_cached_reranked(
        self,
        retrieval_hash: str,
        original_query: str,
    ) -> Optional[list[dict[str, Any]]]:
        """Retrieve cached reranked verses."""
        cache_key = CacheKeys.rerank_key(retrieval_hash, original_query)
        cached = self.cache_manager.get(cache_key)
        if cached and isinstance(cached, dict):
            verses = cached.get("verses", [])
            logger.debug("[SESSION] Hit: %d cached reranked verses", len(verses))
            return verses
        return None

    def get_session_context_summary(self, session_id: str) -> Optional[str]:
        """Get brief summary of session context for LLM."""
        session = self.get_or_create_session(session_id)
        
        if not session.recent_turns:
            return None

        summary_parts = []
        
        if session.detected_themes:
            summary_parts.append(f"Themes: {', '.join(session.detected_themes[:3])}")
        
        if session.detected_emotions:
            summary_parts.append(f"Emotions: {', '.join(session.detected_emotions[:3])}")
        
        if session.recent_turns:
            recent_query = session.recent_turns[-1].user_query
            summary_parts.append(f"Recent: {recent_query[:60]}")

        return " | ".join(summary_parts) if summary_parts else None
