"""Lightweight conversational context resolver for follow-up query rewriting using LangChain.

This module rewrites ambiguous follow-up queries into standalone queries suitable
for routing and retrieval. It uses the small analyzer Groq model for fast,
cheap, deterministic JSON outputs.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Iterable, List

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from .decomposer import GroqJSONClient
from .config import QueryEngineConfig, get_logger

LOGGER = get_logger(__name__)


_TRIGGER_TOKENS = {
    "this",
    "that",
    "it",
    "same",
    "yes",
    "no",
    "why",
    "how",
    "what",
    "where",
    "when",
}


class ContextResolver:
    """Resolve short or ambiguous follow-up queries into standalone queries using LangChain.

    Usage:
        resolver = ContextResolver(groq_client, config)
        resolved = resolver.resolve_if_needed(query, conversation_history)
    """

    def __init__(self, groq_client: GroqJSONClient | None = None, config: QueryEngineConfig | None = None) -> None:
        self.groq_client = groq_client
        self.config = config or QueryEngineConfig()
        self.logger = LOGGER
        self.llm = ChatGroq(
            api_key=self.config.groq_api_key,
            model_name=getattr(self.config, "groq_analyzer_model_name", self.config.groq_model_name),
            temperature=self.config.groq_temperature,
        )

    @staticmethod
    def _is_ambiguous(user_query: str) -> bool:
        if not user_query:
            return True
        words = user_query.strip().split()
        if len(words) < 5:
            return True
        # Check for trigger tokens (word boundaries, case-insensitive)
        tokens = re.findall(r"\b[\w']+\b", user_query.lower())
        for t in tokens:
            if t in _TRIGGER_TOKENS:
                return True
        return False

    def resolve_if_needed(self, user_query: str, conversation_history: List[dict[str, str]] | None) -> str:
        """Return a resolved standalone query, or the original if not needed.

        The resolver runs only when the query appears ambiguous according to
        lightweight trigger rules (short length or contains reference tokens).
        """
        try:
            if not self._is_ambiguous(user_query):
                self.logger.debug("ContextResolver: query is not ambiguous: %s", user_query)
                return user_query

            # Build a small context window (last N turns)
            context_window: List[str] = []
            if conversation_history:
                # Use last up to 6 messages (user+assistant pairs)
                for turn in conversation_history[-6:]:
                    role = turn.get("role", "user")
                    content = (turn.get("content") or "").strip()
                    if content:
                        context_window.append(f"{role.upper()}: {content}")

            prompt = self._build_prompt(user_query, context_window)

            # Use LangChain ChatGroq for JSON response
            messages = [
                SystemMessage(content="Return valid JSON only."),
                HumanMessage(content=prompt),
            ]
            response = self.llm.invoke(messages)
            raw_content = response.content or "{}"
            
            # Parse JSON response
            try:
                result = json.loads(raw_content)
            except json.JSONDecodeError:
                import re
                match = re.search(r"\{.*\}", raw_content, re.DOTALL)
                if not match:
                    raise ValueError("LLM response did not contain a JSON object.")
                result = json.loads(match.group(0))

            if not isinstance(result, dict):
                raise ValueError("Resolver expected JSON object from LLM")

            resolved = result.get("resolved_query")
            if not resolved or not isinstance(resolved, str):
                raise ValueError("Resolver returned invalid resolved_query")

            resolved = resolved.strip()
            if not resolved:
                raise ValueError("Resolved query empty")

            self.logger.info("ContextResolver: resolved '%s' -> '%s'", user_query, resolved)
            return resolved

        except Exception as exc:
            # Always fail gracefully: return original query
            self.logger.warning("ContextResolver failed, returning original query: %s", exc)
            return user_query

    def is_continuation(self, user_query: str) -> bool:
        """Detect continuation-like queries that ask to continue or elaborate.

        These are typically very short tokens or single-word prompts like
        'and?', 'continue', 'elaborate', 'more', 'what else?'.
        """
        if not user_query:
            return False
        q = user_query.strip().lower()
        # short continuations or punctuation-only
        if len(q) <= 3 and re.match(r'^[\?!.]+$|^and$|^and\?$|^\.$', q):
            return True

        # common continuation signals
        continuation_signals = {
            "and",
            "and?",
            "why",
            "how",
            "then",
            "so",
            "continue",
            "elaborate",
            "more",
            "what else",
            "what else?",
            "what more",
            "more?",
        }

        # exact-match short phrases
        if q in continuation_signals:
            return True

        # phrases that start with continuation verbs
        if any(q.startswith(sig) for sig in ("continue", "elaborate", "more", "what else", "then", "so")):
            return True

        # fallback: if query length small and contains only stopwords or punctuation
        tokens = re.findall(r"\b[\w']+\b", q)
        if 0 < len(tokens) <= 2:
            # single-token questions like 'and?', 'why?'
            if any(t in continuation_signals for t in tokens):
                return True

        return False

    def _build_prompt(self, user_query: str, context_lines: Iterable[str]) -> str:
        """Construct a compact prompt that asks the LLM to rewrite the query.

        Prompt returns STRICT JSON: {"resolved_query": "..."}
        """
        instructions = (
            "Rewrite the FOLLOW-UP user query into a concise, standalone query suitable for retrieval.\n"
            "- Preserve the user's intent and emotional tone.\n"
            "- Do NOT add unrelated content or philosophy.\n"
            "- Keep it short and retrieval-friendly.\n"
            "Return STRICT JSON only: {\"resolved_query\": \"...\"}.\n"
        )

        context_text = "\n".join(context_lines) if context_lines else ""

        prompt = (
            f"{instructions}"
            f"CONTEXT:\n{context_text}\n"
            f"FOLLOW-UP QUERY: {user_query}\n"
            f"If the query is already clear, return it unchanged."
        )
        return prompt
