"""Unified problem decomposition and emotion detection in a single efficient LLM call."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from .config import QueryEngineConfig, get_logger
from .decomposer import GroqJSONClient
from .models import CombinedAnalysisResult, EmotionLabel, EmotionResult, Problem, ProblemWithEmotion
from .prompts import render_combined_analysis_prompt

LOGGER = get_logger(__name__)


class CombinedAnalyzer:
    """
    Merges query decomposition and emotion detection into ONE efficient LLM call.

    This replaces the separate decomposer + emotion_detector pipeline,
    reducing token usage and latency significantly.
    """

    def __init__(self, groq_client: GroqJSONClient, config: QueryEngineConfig | None = None) -> None:
        self.groq_client = groq_client
        self.config = config or QueryEngineConfig()
        self.logger = get_logger(self.__class__.__name__)

    def analyze(self, query: str) -> tuple[list[Problem], list[EmotionResult]]:
        """
        Perform unified analysis: decompose query and detect emotions in one call.
        
        OPTIMIZED:
        - Uses smaller model (llama-3.1-8b-instant instead of 70b)
        - Compact, deterministic prompt (token-optimized)
        - Single LLM call instead of separate pipeline
        - Graceful fallback for failures

        Args:
            query: User's query string.

        Returns:
            Tuple of (problems list, emotion_results list) preserving original order.

        Raises:
            ValueError: If analysis fails after retries.
        """
        if not query.strip():
            raise ValueError("Query must not be empty.")

        # Single LLM call for both decomposition and emotion detection
        # OPTIMIZED: Use smaller analyzer model (llama-3.1-8b-instant) for extraction tasks
        payload = self.groq_client.invoke_json(
            render_combined_analysis_prompt(query),
            model_name=self.config.groq_analyzer_model_name,  # Use smaller model for extraction
        )

        try:
            result = CombinedAnalysisResult.model_validate(payload)
        except ValidationError as exc:
            self.logger.error("Invalid combined analysis payload: %s", payload)
            raise ValueError(f"Invalid combined analysis response: {exc}") from exc

        if not result.problems:
            self.logger.warning("Combined analyzer returned no problems for query: %s", query)
            # Fallback: treat as single generic query
            result.problems = [
                ProblemWithEmotion(
                    problem="generic informational query",
                    emotions=["none"],
                )
            ]

        # Convert combined results into separate Problem and EmotionResult lists
        problems = []
        emotion_results = []

        for item in result.problems:
            # Validate and normalize emotions
            normalized_emotions = self._normalize_emotions(item.emotions)

            # Create Problem record
            problems.append(Problem(problem=item.problem))

            # Create EmotionResult for each emotion (or single dominant one)
            # For compatibility with downstream, we pick the first emotion or "none"
            dominant_emotion = normalized_emotions[0] if normalized_emotions else "none"
            emotion_results.append(
                EmotionResult(
                    problem=item.problem,
                    emotion=dominant_emotion,
                )
            )

        self.logger.info(
            "Combined analysis: %d problems identified with emotions",
            len(problems),
        )
        for problem, emotion_result in zip(problems, emotion_results):
            self.logger.debug(
                "Problem: '%s' -> Emotion: '%s'",
                problem.problem,
                emotion_result.emotion,
            )

        return problems, emotion_results

    def _normalize_emotions(self, emotions: list[str]) -> list[EmotionLabel]:
        """
        Validate and normalize emotions from LLM response.

        Args:
            emotions: List of emotion strings from the LLM.

        Returns:
            List of valid EmotionLabel values, up to 3.
        """
        ALLOWED = {
            "fear",
            "confusion",
            "grief",
            "anger",
            "anxiety",
            "attachment",
            "doubt",
            "guilt",
            "loneliness",
            "hopelessness",
            "restlessness",
            "peace",
            "courage",
            "surrender",
            "clarity",
            "discipline",
            "frustration",
            "none",
        }

        normalized = []
        for emotion in emotions:
            cleaned = (emotion or "").strip().lower()
            if cleaned in ALLOWED:
                if cleaned not in normalized:  # Avoid duplicates
                    normalized.append(cleaned)  # type: ignore
            else:
                self.logger.warning(
                    "Unknown emotion '%s' from LLM, using 'none' as fallback",
                    emotion,
                )
                if "none" not in normalized:
                    normalized.append("none")  # type: ignore

        # Cap at 3 emotions
        return normalized[:3]  # type: ignore
