"""Emotion detection for decomposed philosophical concerns."""

from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError

from .decomposer import GroqJSONClient
from .models import EmotionResult, Problem
from .prompts import render_emotion_prompt
from .config import get_logger

LOGGER = get_logger(__name__)


class EmotionEnvelope(BaseModel):
    """Expected LLM payload for batch emotion classification."""

    results: list[EmotionResult] = Field(default_factory=list)


class EmotionDetector:
    """Classify the dominant emotion for each decomposed problem."""

    def __init__(self, groq_client: GroqJSONClient) -> None:
        self.groq_client = groq_client

    def detect(self, problems: list[Problem]) -> list[EmotionResult]:
        if not problems:
            return []

        payload = self.groq_client.invoke_json(
            render_emotion_prompt([problem.problem for problem in problems])
        )
        try:
            envelope = EmotionEnvelope.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(f"Invalid emotion payload: {payload}") from exc

        result_map = {result.problem: result for result in envelope.results}
        ordered_results: list[EmotionResult] = []
        for problem in problems:
            if problem.problem in result_map:
                ordered_results.append(result_map[problem.problem])

        if len(ordered_results) != len(problems):
            raise ValueError(
                "Emotion detector did not return a complete set of results "
                f"for all problems. Expected {len(problems)}, got {len(ordered_results)}."
            )

        LOGGER.info(
            "Detected emotions: %s",
            [{"problem": item.problem, "emotion": item.emotion} for item in ordered_results],
        )
        return ordered_results
