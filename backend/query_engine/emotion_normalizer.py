"""Normalize raw emotions into canonical Bhagavad Gita emotion labels."""

from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError

from .config import get_logger
from .decomposer import GroqJSONClient
from .models import EmotionNormalizationResult
from .prompts import render_emotion_normalization_prompt

LOGGER = get_logger(__name__)


class EmotionNormalizationEnvelope(BaseModel):
    """Expected LLM payload for canonical emotion normalization."""

    mapped_emotions: list[str] = Field(default_factory=list, min_length=1, max_length=3)


class EmotionNormalizer:
    """Map raw emotions onto the system's canonical emotion vocabulary."""

    def __init__(self, groq_client: GroqJSONClient) -> None:
        self.groq_client = groq_client

    def normalize(self, emotion: str) -> EmotionNormalizationResult:
        if not emotion.strip():
            raise ValueError("emotion must not be empty.")

        payload = self.groq_client.invoke_json(
            render_emotion_normalization_prompt(emotion.strip())
        )
        try:
            envelope = EmotionNormalizationEnvelope.model_validate(payload)
            result = EmotionNormalizationResult(
                input_emotion=emotion.strip(),
                mapped_emotions=envelope.mapped_emotions,
            )
        except ValidationError as exc:
            raise ValueError(f"Invalid emotion normalization payload: {payload}") from exc

        LOGGER.info(
            "Normalized emotion '%s' to %s",
            result.input_emotion,
            result.mapped_emotions,
        )
        return result
