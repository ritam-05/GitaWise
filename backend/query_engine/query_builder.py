"""Build optimized dense retrieval queries from problems and emotions.

REFACTORED: Now uses programmatic generation (NO LLM) instead of expensive query-builder LLM call.
This reduces orchestration overhead and token consumption significantly.
"""

from __future__ import annotations

from .config import get_logger
from .models import EmotionResult, RetrievalQuery
from .query_generator import ProgrammaticQueryGenerator

LOGGER = get_logger(__name__)


class RetrievalQueryBuilder:
    """Convert problem-emotion pairs into semantic retrieval queries.
    
    NOW: Uses programmatic generation with semantic enrichment.
    BEFORE: Used an LLM call (expensive, slow, token-heavy).
    
    Zero LLM calls. Deterministic. Fast.
    """

    def __init__(self, groq_client=None) -> None:
        # groq_client parameter kept for backward compatibility but not used
        self.generator = ProgrammaticQueryGenerator()
        LOGGER.info("RetrievalQueryBuilder initialized (programmatic mode, no LLM)")

    def build(self, emotion_results: list[EmotionResult]) -> list[RetrievalQuery]:
        """
        Generate retrieval queries programmatically.
        
        Args:
            emotion_results: List of (problem, emotion) pairs
            
        Returns:
            List of optimized retrieval queries
        """
        if not emotion_results:
            return []

        queries = self.generator.generate(emotion_results)

        LOGGER.info(
            "Generated retrieval queries: %s",
            [
                {"problem": item.problem, "emotion": item.emotion, "query": item.query}
                for item in queries
            ],
        )
        return queries
