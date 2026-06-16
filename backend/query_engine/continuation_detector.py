from __future__ import annotations

import re
from typing import List, Optional


class ContinuationDetector:
    """Rule-based continuation detector for short follow-ups.

    Lightweight and fast — uses simple heuristics. Can be extended to fallback
    to a tiny model if desired.
    """

    # Only very explicit continuation signals — NOT question words like how/why/so
    CONTINUATION_KEYWORDS = {
        "and",
        "and?",
        "then",
        "continue",
        "elaborate",
        "more",
        "what else",
        "what else?",
        "more?",
        "same",
        "same here",
        "it",
        "go on",
        "keep going",
        "next",
        "tell me more",
    }

    def __init__(self, min_words: int = 3) -> None:
        self.min_words = min_words

    def is_continuation(self, user_query: str, conversation_history: Optional[List[dict]] = None) -> bool:
        if not user_query or not user_query.strip():
            return False

        q = user_query.strip().lower()
        tokens = re.findall(r"\b[\w']+\b", q)

        # Very short queries (1-2 tokens) that are explicit continuation words
        if len(tokens) < self.min_words:
            if len(tokens) == 0:
                return True
            if any(t in self.CONTINUATION_KEYWORDS for t in tokens):
                return True
            # single-token queries that are not question words or substantive content
            question_words = {"why", "how", "what", "where", "when", "who", "which"}
            if len(tokens) == 1 and tokens[0] not in question_words:
                return True

        # Exact keyword match or phrase match only (not substring — avoids "how" matching "how to deal...")
        for sig in self.CONTINUATION_KEYWORDS:
            if q == sig or q == sig.rstrip("?"):
                return True

        return False
