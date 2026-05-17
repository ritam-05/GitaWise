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
    
    # Recent conversation context (max_length removed - enforced by add_turn)
    recent_turns: list[ConversationTurn] = Field(default_factory=list)
    
    # Themes and patterns
    detected_themes: list[str] = Field(default_factory=list)
    detected_emotions: list[str] = Field(default_factory=list)
    dominant_route: Optional[str] = None
    # Last known route/topic/resolved query/retrieval for continuation semantics
    last_route: Optional[str] = None
    last_topic: Optional[str] = None
    last_resolved_query: Optional[str] = None
    last_retrieved_verses: list[dict[str, Any]] = Field(default_factory=list)
    # Store resolved follow-up rewrites for debugging and continuity
    resolved_followups: list[dict[str, str]] = Field(default_factory=list)
    # Conversation meta state
    conversation_state: Optional[str] = None
    last_transition_type: Optional[str] = None
    
    # Statistics
    total_turns: int = 0
    total_verses_retrieved: int = 0

    def add_turn(self, turn: ConversationTurn, max_stored_turns: int = 10) -> None:
        """
        Add conversation turn to memory with configurable limit.
        
        Args:
            turn: ConversationTurn to add
            max_stored_turns: Maximum turns to keep in memory (default: 10)
        """
        self.recent_turns.append(turn)
        # Enforce max limit, removing oldest turns if needed
        if len(self.recent_turns) > max_stored_turns:
            excess = len(self.recent_turns) - max_stored_turns
            self.recent_turns = self.recent_turns[excess:]
        self.total_turns += 1
        self.last_accessed = datetime.utcnow()
        # Update last_topic/last_route/last_resolved_query when a turn is tracked
        try:
            # turn.route may be present
            if hasattr(turn, 'route') and turn.route:
                self.last_route = turn.route
            # inferred topic from problems if present
            if turn.problems:
                first_problem = turn.problems[0]
                if isinstance(first_problem, dict):
                    self.last_topic = first_problem.get('problem') or self.last_topic
            # last_resolved_query is not available here by default; engine should set it explicitly
        except Exception:
            pass

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

    def __init__(
        self,
        cache_manager: CacheManager,
        session_ttl_seconds: int = 3600,
        session_max_stored_turns: int = 10,
        session_context_turns: int = 5,
        session_cleanup_interval_seconds: int = 1800,
    ) -> None:
        """
        Initialize session cache.
        
        Args:
            cache_manager: Global cache manager instance
            session_ttl_seconds: Session TTL before expiration (default: 1 hour)
            session_max_stored_turns: Maximum turns to store in memory (default: 10)
            session_context_turns: Number of turns to send to LLM (default: 5)
            session_cleanup_interval_seconds: Cleanup task interval (default: 30 min)
        """
        self.cache_manager = cache_manager
        self.session_ttl_seconds = session_ttl_seconds
        self.session_max_stored_turns = session_max_stored_turns
        self.session_context_turns = session_context_turns
        self.session_cleanup_interval_seconds = session_cleanup_interval_seconds
        self._last_cleanup = datetime.utcnow()
        
        logger.info(
            "[SESSION_CACHE] Initialized (ttl=%ds, max_turns=%d, context_turns=%d, cleanup_interval=%ds)",
            session_ttl_seconds,
            session_max_stored_turns,
            session_context_turns,
            session_cleanup_interval_seconds,
        )

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

    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions from cache.
        
        Runs periodically to prevent memory bloat.
        
        Returns:
            Number of sessions cleaned up
        """
        now = datetime.utcnow()
        time_since_last_cleanup = (now - self._last_cleanup).total_seconds()
        
        # Check if cleanup interval has passed
        if time_since_last_cleanup < self.session_cleanup_interval_seconds:
            logger.debug(
                "[SESSION_CLEANUP] Skipping cleanup (%.0f seconds since last, interval: %d)",
                time_since_last_cleanup,
                self.session_cleanup_interval_seconds,
            )
            return 0
        
        self._last_cleanup = now
        cleaned_count = 0
        
        try:
            # Note: This is a best-effort cleanup for in-memory and Redis caches.
            # For in-memory cache, we scan all keys; for Redis, TTL is handled automatically.
            # For production, consider implementing explicit session tracking.
            logger.info(
                "[SESSION_CLEANUP] Starting cleanup (ttl=%ds, threshold_time=%s)",
                self.session_ttl_seconds,
                (now - timedelta(seconds=self.session_ttl_seconds)).isoformat(),
            )
            
            # If cache_manager has a way to list all session keys, we'd iterate and check expiry
            # For now, log the cleanup action for monitoring
            logger.info("[SESSION_CLEANUP] Cleanup cycle completed (cleaned: %d sessions)", cleaned_count)
            
        except Exception as exc:
            logger.warning("[SESSION_CLEANUP] Cleanup failed: %s", exc)
        
        return cleaned_count

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

    def record_resolved_followup(self, session_id: str, original: str, resolved: str, max_entries: int = 20) -> None:
        """Record a resolved follow-up query in the session memory.

        Keeps a short history of original -> resolved mappings to help
        subsequent turns use the clarified query.
        """
        try:
            session = self.get_or_create_session(session_id)
            entry = {"original": original, "resolved": resolved, "timestamp": datetime.utcnow().isoformat()}
            session.resolved_followups.append(entry)
            # trim to most recent
            if len(session.resolved_followups) > max_entries:
                session.resolved_followups = session.resolved_followups[-max_entries:]
            self.save_session(session)
            logger.debug("[SESSION] Recorded resolved followup for session %s: %s -> %s", session_id, original[:60], resolved[:60])
        except Exception as exc:
            logger.warning("[SESSION] Failed to record resolved followup: %s", exc)

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
