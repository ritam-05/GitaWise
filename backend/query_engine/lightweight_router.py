"""Lightweight rules-based router with NO LLM calls for instant query classification."""

from __future__ import annotations

from difflib import get_close_matches
import re
from .config import get_logger
from .models import RouteLabel, RouteResult

LOGGER = get_logger(__name__)


class LightweightRouter:
    """
    Rules-based query router using regex patterns and heuristics.
    
    ZERO LLM calls. Instant route classification.
    
    Routes:
    - generic_chat: greetings, small talk, non-philosophical
    - philosophical_guidance: life struggles, relationships, discipline, purpose
    - emotion_guidance: intense emotional distress (grief, hopelessness, anxiety)
    - gita_rag: explicit Gita/scripture references
    """

    # Explicit Gita/scripture references (highest priority)
    GITA_PATTERNS = [
        r"\b(bhagavad\s*gita|gita|bhagwad\s*gita)\b",
        r"\b(krishna|arjuna)\b",
        r"\b(chapter|verse|shloka)\b",
        r"\b(karma\s*yoga|bhakti\s*yoga|jnana\s*yoga)\b",
        r"\b(dharma|moksha|brahman)\b",
        r"\bupanishad\b",
    ]

    # Intense emotional distress (grief, hopelessness, loneliness)
    EMOTION_GUIDANCE_PATTERNS = [
        r"\b(grief|grieving|bereaved)\b",
        r"\b(hopeless|hopelessness|suicidal|despair)\b",
        r"\b(lonely|loneliness|isolated|alone)\b",
        r"\b(severely|deeply|intensely).{0,20}(sad|depressed|anxious|suffering)\b",
        r"\b(can't|cannot|unable).{0,20}(cope|bear|handle)\b",
        r"\b(dark|darkness|black\s+hole).{0,20}(thought|mind|feeling)\b",
    ]

    # Philosophical struggle patterns (medium priority)
    PHILOSOPHICAL_PATTERNS = [
        r"\b(purpose|meaning|life|calling)\b",
        r"\b(duty|responsibility|obligation)\b",
        r"\b(discipline|focus|distraction)\b",
        r"\b(attachment|detachment|desire)\b",
        r"\b(fear|anxiety|doubt|confusion)\b",
        r"\b(relationship|love|family)\b",
        r"\b(work|career|job|profession)\b",
        r"\b(struggle|conflict|inner|balance)\b",
        r"\b(wisdom|knowledge|understanding)\b",
        r"\b(action|act|do|doing)\b",
        r"\b(self|soul|spirit|consciousness)\b",
        r"\b(meditat|pray|reflect|contemplate)\b",
    ]

    # Generic chat patterns (lowest priority - greetings, casual)
    GENERIC_PATTERNS = [
        r"^(hi|hello|hey|greetings|thanks|thank\s+you|okay|ok|yes|no)\b",
        r"\b(weather|time|date|how\s+are\s+you)\b",
        r"\b(joke|funny|laugh)\b",
        r"\b(help|question|how\s+to|what\s+is)\b",
    ]

    SHORT_POSITIVE_WORDS = {
        "good",
        "great",
        "nice",
        "ok",
        "okay",
        "thanks",
        "thankyou",
        "perfect",
        "gotit",
        "understood",
        "goof",
        "goood",
        "gd",
        "thumbsup",
    }
    SHORT_NEGATIVE_WORDS = {
        "bad",
        "wrong",
        "no",
        "irrelevant",
        "unclear",
    }
    SHORT_CONTINUATION_WORDS = {
        "and",
        "then",
        "continue",
        "more",
        "next",
        "elaborate",
        "why",
        "how",
    }
    GITA_CONCEPT_WORDS = {
        "dharma",
        "karma",
        "moksha",
        "brahman",
        "atman",
        "krishna",
        "arjuna",
        "gita",
        "bhakti",
        "jnana",
        "yoga",
    }

    def route(self, query: str) -> RouteResult:
        """
        Route a user query to the most appropriate handler.
        
        Priority:
        1. Explicit Gita references → gita_rag
        2. Intense emotional distress → emotion_guidance
        3. Philosophical struggle → philosophical_guidance
        4. Default → generic_chat
        
        Args:
            query: User's input query
            
        Returns:
            RouteResult with selected route
        """
        if not query.strip():
            LOGGER.warning("Empty query received, defaulting to generic_chat")
            return RouteResult(route="generic_chat")

        query_lower = query.lower().strip()
        
        # Rule 1: Explicit Gita references (highest priority)
        if self._matches_patterns(query_lower, self.GITA_PATTERNS):
            LOGGER.info("Routed to 'gita_rag' (explicit scripture reference)")
            return RouteResult(route="gita_rag")
        
        # Rule 2: Intense emotional distress (second priority)
        if self._matches_patterns(query_lower, self.EMOTION_GUIDANCE_PATTERNS):
            LOGGER.info("Routed to 'emotion_guidance' (intense emotional distress detected)")
            return RouteResult(route="emotion_guidance")
        
        # Rule 3: Philosophical struggle (third priority)
        philosophical_match_count = self._count_pattern_matches(query_lower, self.PHILOSOPHICAL_PATTERNS)
        if philosophical_match_count >= 2:  # At least 2 philosophical terms
            LOGGER.info("Routed to 'philosophical_guidance' (%d philosophical terms detected)", philosophical_match_count)
            return RouteResult(route="philosophical_guidance")
        
        if philosophical_match_count == 1 and not self._matches_patterns(query_lower, self.GENERIC_PATTERNS):
            # Single philosophical term with no generic chat markers
            LOGGER.info("Routed to 'philosophical_guidance' (1 philosophical term, no generic markers)")
            return RouteResult(route="philosophical_guidance")
        
        # Rule 4: Default to generic_chat
        LOGGER.info("Routed to 'generic_chat' (default classification)")
        return RouteResult(route="generic_chat")

    @classmethod
    def fuzzy_match_short_token(cls, token: str, candidates: set[str], cutoff: float = 0.6) -> str | None:
        """Return the closest short-token match for typo-tolerant intent detection."""
        normalized = cls.normalize_short_text(token)
        if not normalized:
            return None
        if normalized in candidates:
            return normalized
        matches = get_close_matches(normalized, list(candidates), n=1, cutoff=cutoff)
        if not matches:
            return None
        match = matches[0]
        if abs(len(match) - len(normalized)) <= 2:
            return match
        return None

    @classmethod
    def fuzzy_match_text(cls, text: str, candidates: set[str], cutoff: float = 0.72) -> str | None:
        """Return the closest normalized text match for short intent phrases."""
        normalized = cls.normalize_phrase(text)
        normalized_candidates = {cls.normalize_phrase(candidate): candidate for candidate in candidates}
        if not normalized:
            return None
        if normalized in normalized_candidates:
            return normalized_candidates[normalized]
        matches = get_close_matches(normalized, list(normalized_candidates.keys()), n=1, cutoff=cutoff)
        if not matches:
            return None
        return normalized_candidates[matches[0]]

    @staticmethod
    def normalize_short_text(text: str) -> str:
        """Normalize a short phrase for fuzzy feedback matching."""
        return re.sub(r"[^a-z0-9]+", "", text.lower()).strip()

    @staticmethod
    def normalize_phrase(text: str) -> str:
        """Normalize a phrase while preserving word boundaries."""
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]+", " ", text.lower())).strip()

    @classmethod
    def is_gita_concept_word(cls, token: str) -> bool:
        return cls.normalize_short_text(token) in cls.GITA_CONCEPT_WORDS

    @staticmethod
    def _matches_patterns(text: str, patterns: list[str]) -> bool:
        """Check if text matches any of the regex patterns."""
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)

    @staticmethod
    def _count_pattern_matches(text: str, patterns: list[str]) -> int:
        """Count how many patterns match in the text."""
        return sum(1 for pattern in patterns if re.search(pattern, text, re.IGNORECASE))
