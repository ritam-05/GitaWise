from __future__ import annotations

import logging
from typing import List, Optional

from .context_resolver import ContextResolver
from .dialogue_state import DialogueState
from .decomposer import GroqJSONClient
from .config import QueryEngineConfig, get_logger

LOGGER = get_logger(__name__)


class ContextualRewriter:
    """Rewrite continuation-style queries into standalone, retrieval-friendly queries using LangChain.

    Uses rule-based composition first; optionally calls the small analyzer
    resolver for improved rewrites when available.
    """

    def __init__(self, groq_client: GroqJSONClient | None = None, config: QueryEngineConfig | None = None) -> None:
        self.groq_client = groq_client
        self.config = config or QueryEngineConfig()
        try:
            self.resolver = ContextResolver(groq_client, self.config)
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
            style_instruction = self._extract_style_instruction(user_query)
            if transition_type == "emotion_shift":
                # Create a sensitive, grounding-focused rewrite that does not force prior topic
                if q:
                    resolved = f"User expresses acute emotional distress: {q}. Provide calm, grounded guidance referencing suffering and consolation in the Bhagavad Gita."
                    return self._with_style_instruction(resolved, style_instruction)
                return "User expresses acute emotional distress. Provide calm, grounded guidance referencing suffering and consolation in the Bhagavad Gita."

            # If we have a clear active topic, prefer a deterministic composition
            if state and state.active_topic:
                base = state.active_topic
                # preserve user intent signals like 'how', 'why' when present
                if q.startswith("how"):
                    resolved = f"How can one {base} according to the Bhagavad Gita"
                    return self._with_style_instruction(resolved, style_instruction)
                if q.startswith("why"):
                    resolved = f"Why does the Bhagavad Gita advise about {base}"
                    return self._with_style_instruction(resolved, style_instruction)
                # default continuation composition
                resolved = f"Continue explaining Bhagavad Gita teachings about {base}"
                return self._with_style_instruction(resolved, style_instruction)

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
                return self._with_style_instruction(resolved, style_instruction)

            # Fallback simple rewrite
            resolved = f"Continue: {user_query.strip()}"
            return self._with_style_instruction(resolved, style_instruction)

        except Exception as exc:
            self.logger.warning("ContextualRewriter failed, falling back to original query: %s", exc)
            return user_query

    @staticmethod
    def _extract_style_instruction(user_query: str) -> str:
        q = (user_query or "").strip().lower()
        if not q:
            return ""

        style_markers = [
            ("bullet", "Answer in bullet points."),
            ("point wise", "Answer in bullet points."),
            ("point-wise", "Answer in bullet points."),
            ("numbered", "Answer as a numbered list."),
            ("step by step", "Answer step by step."),
            ("step-by-step", "Answer step by step."),
            ("table", "Answer in a table."),
            ("short", "Keep the answer concise."),
            ("detailed", "Give a detailed answer."),
        ]
        instructions = []
        for marker, instruction in style_markers:
            if marker in q and instruction not in instructions:
                instructions.append(instruction)
        return " ".join(instructions)

    @staticmethod
    def _with_style_instruction(resolved_query: str, style_instruction: str) -> str:
        if not style_instruction:
            return resolved_query
        return f"{resolved_query}. Current user style request: {style_instruction}"
