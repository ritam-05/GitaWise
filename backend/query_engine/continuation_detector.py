from __future__ import annotations

import re
from typing import List, Optional


class ContinuationDetector:
    """Rule-based continuation detector for short follow-ups.

    Lightweight and fast — uses simple heuristics. Can be extended to fallback
    to a tiny model if desired.
    """

    CONTINUATION_KEYWORDS = {
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
        "more?",
        "same",
        "same here",
        "it",
    }

    def __init__(self, min_words: int = 4) -> None:
        self.min_words = min_words

    def is_continuation(self, user_query: str, conversation_history: Optional[List[dict]] = None) -> bool:
        if not user_query or not user_query.strip():
            return False

        q = user_query.strip().lower()
        # Very short queries are likely continuations
        tokens = re.findall(r"\b[\w']+\b", q)
        if len(tokens) < self.min_words:
            # If it's a single substantive token like 'why' or punctuation-only
            if len(tokens) == 0:
                return True
            if any(t in self.CONTINUATION_KEYWORDS for t in tokens):
                return True
            # also treat single-token questions as continuations
            if len(tokens) == 1:
                return True

        # Keyword-based detection
        for sig in self.CONTINUATION_KEYWORDS:
            if q == sig or q.startswith(sig + " ") or q.endswith(" " + sig) or sig in q:
                return True

        return False
