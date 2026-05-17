"""LLM-backed query decomposition for philosophical and emotional concerns."""

from __future__ import annotations

import json
import re
from urllib import error, request
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


class SarvamTextClient:
    """Thin helper around the Sarvam chat completions API for text generation."""

    ENDPOINT = "https://api.sarvam.ai/v1/chat/completions"

    def __init__(self, config: QueryEngineConfig) -> None:
        self.config = config

    def invoke_text(self, prompt: str, model_name: str | None = None) -> str:
        """
        Invoke Sarvam for text response generation.

        Args:
            prompt: The prompt to send.
            model_name: Optional model override. Uses config.sarvam_model_name if not provided.
        """
        primary_model = model_name or self.config.sarvam_model_name
        fallback_model = getattr(self.config, "sarvam_fallback_model_name", "sarvam-m")
        models_to_try: list[str] = [primary_model]
        if fallback_model and fallback_model not in models_to_try:
            models_to_try.append(fallback_model)

        last_error: Exception | None = None
        for candidate_model in models_to_try:
            for attempt in range(1, self.config.sarvam_max_retries + 2):
                try:
                    payload = self._build_payload(prompt, candidate_model)
                    req = request.Request(
                        self.ENDPOINT,
                        data=json.dumps(payload).encode("utf-8"),
                        headers=self._headers(),
                        method="POST",
                    )
                    with request.urlopen(req, timeout=self.config.sarvam_timeout_seconds) as response:
                        raw_body = response.read().decode("utf-8")

                    parsed = json.loads(raw_body)
                    return self._extract_text(parsed)
                except (error.HTTPError, error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
                    last_error = exc
                    LOGGER.warning(
                        "Sarvam text attempt %s failed for model %s: %s",
                        attempt,
                        candidate_model,
                        exc,
                    )

        raise RuntimeError("Sarvam text invocation failed after retries.") from last_error

    def _build_payload(self, prompt: str, model: str) -> dict[str, Any]:
        return {
            "model": model,
            "temperature": self.config.sarvam_temperature,
            "max_tokens": self.config.sarvam_max_tokens,
            "reasoning_effort": None,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Respond carefully and only from the provided Bhagavad Gita context when "
                        "context is supplied. Return only the final answer."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "api-subscription-key": self.config.sarvam_api_key,
        }

    @staticmethod
    def _extract_text(payload: dict[str, Any]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("Sarvam response did not include any choices.")

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise ValueError("Sarvam response choice had an unexpected shape.")

        message = first_choice.get("message", {})
        if not isinstance(message, dict):
            raise ValueError("Sarvam response message had an unexpected shape.")

        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Sarvam response did not include text content.")

        cleaned = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL | re.IGNORECASE).strip()
        cleaned = cleaned.strip('"').strip()
        if not cleaned:
            raise ValueError("Sarvam response content was empty after sanitization.")
        return cleaned
