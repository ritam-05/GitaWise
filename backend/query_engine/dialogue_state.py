from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class DialogueState:
    """Structured dialogue state used for continuation and context inheritance."""

    active_topic: Optional[str] = None
    active_route: Optional[str] = None
    active_emotions: List[str] = field(default_factory=list)
    active_intent: Optional[str] = None
    last_resolved_query: Optional[str] = None
    last_retrieved_verses: List[dict[str, Any]] = field(default_factory=list)
    last_philosophical_theme: Optional[str] = None
    # Meta conversation state (continuation|topic_shift|emotion_shift)
    conversation_state: Optional[str] = None
    last_transition_type: Optional[str] = None

    @classmethod
    def from_session(cls, session: object) -> "DialogueState":
        if not session:
            return cls()

        return cls(
            active_topic=getattr(session, "last_topic", None),
            active_route=getattr(session, "last_route", None),
            active_emotions=getattr(session, "detected_emotions", []) or [],
            active_intent=getattr(session, "last_resolved_query", None),
            last_resolved_query=getattr(session, "last_resolved_query", None),
            last_retrieved_verses=getattr(session, "last_retrieved_verses", []) or [],
            last_philosophical_theme=(getattr(session, "detected_themes", []) or [None])[0],
            conversation_state=getattr(session, "conversation_state", None),
            last_transition_type=getattr(session, "last_transition_type", None),
        )

    def apply_to_session(self, session: object) -> None:
        """Persist the structured dialogue state back into the session object.

        This keeps the session as the single source of truth for continuity.
        """
        if not session:
            return

        try:
            if hasattr(session, "last_topic"):
                session.last_topic = self.active_topic
            if hasattr(session, "last_route"):
                session.last_route = self.active_route
            if hasattr(session, "last_resolved_query"):
                session.last_resolved_query = self.last_resolved_query
            if hasattr(session, "last_retrieved_verses"):
                session.last_retrieved_verses = self.last_retrieved_verses
            if hasattr(session, "detected_themes") and self.last_philosophical_theme:
                themes = list({self.last_philosophical_theme} | set(getattr(session, "detected_themes", []) or []))
                session.detected_themes = themes[:5]
            if hasattr(session, "detected_emotions"):
                session.detected_emotions = list(self.active_emotions)[:8]
            if hasattr(session, "conversation_state"):
                session.conversation_state = self.conversation_state
            if hasattr(session, "last_transition_type"):
                session.last_transition_type = self.last_transition_type
        except Exception:
            # best-effort only; don't raise
            return
