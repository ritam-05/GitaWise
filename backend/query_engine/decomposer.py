"""LLM-backed query decomposition for philosophical and emotional concerns."""

from __future__ import annotations

import json
import re
from typing import Any

from groq import Groq
from pydantic import BaseModel, Field, ValidationError

from .config import QueryEngineConfig, get_logger
from .models import Problem

LOGGER = get_logger(__name__)


class GroqJSONClient:
    """Thin helper around the Groq chat completion API for strict JSON calls."""

    def __init__(self, config: QueryEngineConfig) -> None:
        self.config = config
        self.client = Groq(api_key=config.groq_api_key)

    def invoke_json(self, prompt: str, model_name: str | None = None) -> dict[str, Any]:
        """
        Invoke Groq with JSON response format.
        
        Args:
            prompt: The prompt to send
            model_name: Optional model override. Uses config.groq_model_name if not provided.
        """
        model = model_name or self.config.groq_model_name
        last_error: Exception | None = None
        for attempt in range(1, self.config.groq_max_retries + 2):
            try:
                completion = self.client.chat.completions.create(
                    model=model,
                    temperature=self.config.groq_temperature,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": "Return valid JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                )
                raw_content = completion.choices[0].message.content or "{}"
                return self._extract_json(raw_content)
            except Exception as exc:
                last_error = exc
                LOGGER.warning("Groq JSON attempt %s failed: %s", attempt, exc)

        raise RuntimeError("Groq JSON invocation failed after retries.") from last_error

    def invoke_text(self, prompt: str, model_name: str | None = None) -> str:
        """
        Invoke Groq with text response.
        
        Args:
            prompt: The prompt to send
            model_name: Optional model override. Uses config.groq_model_name if not provided.
        """
        model = model_name or self.config.groq_model_name
        last_error: Exception | None = None
        for attempt in range(1, self.config.groq_max_retries + 2):
            try:
                completion = self.client.chat.completions.create(
                    model=model,
                    temperature=self.config.groq_temperature,
                    messages=[
                        {
                            "role": "system",
                            "content": "Respond carefully and only from the provided Bhagavad Gita context.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                )
                return (completion.choices[0].message.content or "").strip()
            except Exception as exc:
                last_error = exc
                LOGGER.warning("Groq text attempt %s failed: %s", attempt, exc)

        raise RuntimeError("Groq text invocation failed after retries.") from last_error

    @staticmethod
    def _extract_json(raw_text: str) -> dict[str, Any]:
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if not match:
                raise ValueError("LLM response did not contain a JSON object.")
            return json.loads(match.group(0))
