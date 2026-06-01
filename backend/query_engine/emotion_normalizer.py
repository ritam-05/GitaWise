"""Normalize raw emotions into canonical Bhagavad Gita emotion labels using LangChain."""

from __future__ import annotations

import json
from pydantic import BaseModel, Field, ValidationError

from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, SystemMessage

from .config import QueryEngineConfig, get_logger
from .decomposer import GroqJSONClient
from .models import EmotionNormalizationResult
from .prompts import render_emotion_normalization_prompt, ALLOWED_EMOTIONS

LOGGER = get_logger(__name__)


class EmotionNormalizationEnvelope(BaseModel):
    """Expected LLM payload for canonical emotion normalization."""

    mapped_emotions: list[str] = Field(default_factory=list, min_length=1, max_length=3)


class EmotionNormalizer:
    """Map raw emotions onto the system's canonical emotion vocabulary using LangChain."""

    def __init__(self, groq_client: GroqJSONClient | None = None, config: QueryEngineConfig | None = None) -> None:
        self.groq_client = groq_client
        self.config = config or QueryEngineConfig()
        self.llm = ChatGroq(
            api_key=self.config.groq_api_key,
            model_name=self.config.groq_model_name,
            temperature=self.config.groq_temperature,
        )

    def normalize(self, emotion: str) -> EmotionNormalizationResult:
        if not emotion.strip():
            raise ValueError("emotion must not be empty.")

        # Use LangChain chain with prompt template, LLM, and JSON parser
        prompt_text = render_emotion_normalization_prompt(emotion.strip())
        
        messages = [
            SystemMessage(content="Return valid JSON only."),
            HumanMessage(content=prompt_text),
        ]
        
        response = self.llm.invoke(messages)
        raw_content = response.content or "{}"
        
        # Parse JSON response
        try:
            payload = json.loads(raw_content)
        except json.JSONDecodeError:
            import re
            match = re.search(r"\{.*\}", raw_content, re.DOTALL)
            if not match:
                raise ValueError("LLM response did not contain a JSON object.")
            payload = json.loads(match.group(0))
        
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
