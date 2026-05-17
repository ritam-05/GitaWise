from __future__ import annotations

import logging
from typing import List, Optional

from .context_resolver import ContextResolver
from .dialogue_state import DialogueState
from .decomposer import GroqJSONClient
from .config import QueryEngineConfig, get_logger

LOGGER = get_logger(__name__)


class ContextualRewriter:
    """Rewrite continuation-style queries into standalone, retrieval-friendly queries.

    Uses rule-based composition first; optionally calls the small analyzer
    resolver for improved rewrites when available.
    """

    def __init__(self, groq_client: GroqJSONClient, config: QueryEngineConfig) -> None:
        self.groq_client = groq_client
        self.config = config
        try:
            self.resolver = ContextResolver(groq_client, config)
        except Exception:
            self.resolver = None
        self.logger = LOGGER

    def rewrite(
        self,
        user_query: str,
        state: DialogueState,
        conversation_history: Optional[List[dict]] = None,
        transition_type: Optional[str] = None,
    ) -> str:
        """Return a concise standalone query derived from the dialogue state.

        Falls back to a compact rule-based composition if the resolver is unavailable.
        """
        try:
            # If emotional escalation, avoid inheriting topic blindly
            q = user_query.strip().lower()
            if transition_type == "emotion_shift":
                # Create a sensitive, grounding-focused rewrite that does not force prior topic
                if q:
                    return f"User expresses acute emotional distress: {q}. Provide calm, grounded guidance referencing suffering and consolation in the Bhagavad Gita."
                return "User expresses acute emotional distress. Provide calm, grounded guidance referencing suffering and consolation in the Bhagavad Gita."

            # If we have a clear active topic, prefer a deterministic composition
            if state and state.active_topic:
                base = state.active_topic
                # preserve user intent signals like 'how', 'why' when present
                if q.startswith("how"):
                    return f"How can one {base} according to the Bhagavad Gita"
                if q.startswith("why"):
                    return f"Why does the Bhagavad Gita advise about {base}"
                # default continuation composition
                return f"Continue explaining Bhagavad Gita teachings about {base}"

            # If no clear topic, try to enrich using the resolver with context
            if self.resolver:
                # Build a small conversation context as list of dicts
                ctx: list[dict[str, str]] = []
                if state and state.last_resolved_query:
                    ctx.append({"role": "system", "content": f"LAST_RESOLVED: {state.last_resolved_query}"})
                if conversation_history:
                    for turn in conversation_history[-6:]:
                        role = turn.get("role", "user")
                        content = (turn.get("content") or "").strip()
                        if content:
                            ctx.append({"role": role, "content": content})
                resolved = self.resolver.resolve_if_needed(user_query, ctx)
                return resolved

            # Fallback simple rewrite
            return f"Continue: {user_query.strip()}"

        except Exception as exc:
            self.logger.warning("ContextualRewriter failed, falling back to original query: %s", exc)
            return user_query
