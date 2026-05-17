"""Programmatic retrieval query generation with semantic enrichment (no LLM)."""

from __future__ import annotations

from .config import get_logger
from .models import EmotionLabel, EmotionResult, RetrievalQuery

LOGGER = get_logger(__name__)


class ProgrammaticQueryGenerator:
    """
    Generate retrieval queries programmatically from problems and emotions.
    
    Uses semantic enrichment (keyword expansion) instead of LLM.
    Fast, deterministic, low-token cost.
    """

    # Gita-relevant semantic enrichment keywords by emotion
    EMOTION_KEYWORDS: dict[EmotionLabel, list[str]] = {
        "fear": ["courage", "steadiness", "action", "duty"],
        "confusion": ["clarity", "knowledge", "wisdom", "understanding"],
        "grief": ["acceptance", "surrender", "detachment", "peace"],
        "anger": ["discipline", "control", "equanimity", "steadiness"],
        "anxiety": ["steadiness", "trust", "surrender", "action"],
        "attachment": ["detachment", "sacrifice", "surrender", "non-attachment"],
        "doubt": ["clarity", "knowledge", "faith", "conviction"],
        "guilt": ["action", "duty", "karma", "forgiveness"],
        "loneliness": ["connection", "self", "soul", "unity"],
        "hopelessness": ["faith", "purpose", "meaning", "courage"],
        "restlessness": ["discipline", "focus", "meditation", "steadiness"],
        "peace": ["meditation", "surrender", "wisdom", "balance"],
        "courage": ["action", "duty", "dharma", "strength"],
        "surrender": ["acceptance", "faith", "trust", "detachment"],
        "clarity": ["knowledge", "wisdom", "understanding", "truth"],
        "discipline": ["focus", "action", "duty", "steadiness"],
        "frustration": ["acceptance", "action", "detachment", "steadiness"],
        "none": ["wisdom", "knowledge", "truth", "understanding"],
    }

    def generate(self, emotion_results: list[EmotionResult]) -> list[RetrievalQuery]:
        """
        Generate retrieval queries programmatically.
        
        Args:
            emotion_results: List of (problem, emotion) pairs from analyzer
            
        Returns:
            List of RetrievalQuery objects optimized for embedding search
        """
        if not emotion_results:
            LOGGER.warning("No emotion results provided for query generation")
            return []

        queries: list[RetrievalQuery] = []

        for result in emotion_results:
            # Generate enriched query
            enriched_query = self._generate_query(result.problem, result.emotion)
            
            query_obj = RetrievalQuery(
                problem=result.problem,
                emotion=result.emotion,
                query=enriched_query,
            )
            queries.append(query_obj)
            
            LOGGER.debug(
                "Generated query: problem='%s', emotion='%s', query='%s'",
                result.problem,
                result.emotion,
                enriched_query,
            )

        LOGGER.info(
            "Generated %d retrieval queries",
            len(queries),
        )
        return queries

    @staticmethod
    def _generate_query(problem: str, emotion: EmotionLabel) -> str:
        """
        Generate a single retrieval query by combining problem and emotion keywords.
        
        Args:
            problem: The identified philosophical concern
            emotion: The detected emotion
            
        Returns:
            Semantic query string for dense embedding search
        """
        # Clean problem text
        problem_text = problem.strip()
        
        # Handle "none" emotion (informational queries)
        if emotion == "none":
            # Just return problem as-is for informational queries
            return problem_text
        
        # Get emotion-relevant keywords
        keywords = ProgrammaticQueryGenerator.EMOTION_KEYWORDS.get(emotion, ["wisdom", "knowledge"])
        
        # Select top 2 most relevant keywords
        selected_keywords = keywords[:2]
        
        # Combine: problem + emotion + keywords
        enriched_query = f"{problem_text} {emotion} {' '.join(selected_keywords)}"
        
        # Clean and normalize
        enriched_query = " ".join(enriched_query.split()).lower()
        
        return enriched_query
