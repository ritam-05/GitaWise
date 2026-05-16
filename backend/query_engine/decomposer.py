"""LLM-backed query decomposition for philosophical and emotional concerns."""

from __future__ import annotations

import json
import re
from typing import Any

from groq import Groq
from pydantic import BaseModel, Field, ValidationError

from .config import QueryEngineConfig, get_logger
from .models import Problem
from .prompts import render_decomposition_prompt

LOGGER = get_logger(__name__)


class DecompositionEnvelope(BaseModel):
    """Expected LLM payload for decomposition."""

    problems: list[Problem] = Field(default_factory=list)


class GroqJSONClient:
    """Thin helper around the Groq chat completion API for strict JSON calls."""

    def __init__(self, config: QueryEngineConfig) -> None:
        self.config = config
        self.client = Groq(api_key=config.groq_api_key)

    def invoke_json(self, prompt: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.config.groq_max_retries + 2):
            try:
                completion = self.client.chat.completions.create(
                    model=self.config.groq_model_name,
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

    def invoke_text(self, prompt: str) -> str:
        last_error: Exception | None = None
        for attempt in range(1, self.config.groq_max_retries + 2):
            try:
                completion = self.client.chat.completions.create(
                    model=self.config.groq_model_name,
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


class QueryDecomposer:
    """Decompose a user query into distinct retrieval-worthy problems."""

    def __init__(self, groq_client: GroqJSONClient) -> None:
        self.groq_client = groq_client

    def decompose(self, query: str) -> list[Problem]:
        if not query.strip():
            raise ValueError("Cannot decompose an empty query.")

        payload = self.groq_client.invoke_json(render_decomposition_prompt(query.strip()))
        try:
            envelope = DecompositionEnvelope.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(f"Invalid decomposition payload: {payload}") from exc

        unique_problems: list[Problem] = []
        seen: set[str] = set()
        for item in envelope.problems:
            normalized = item.problem.strip().lower()
            if normalized and normalized not in seen:
                unique_problems.append(Problem(problem=item.problem.strip()))
                seen.add(normalized)

        if not unique_problems:
            raise ValueError("Query decomposition produced no usable problems.")

        LOGGER.info("Detected problems: %s", [problem.problem for problem in unique_problems])
        return unique_problems
