"""Adaptive route selection for incoming user queries."""

from __future__ import annotations

from pydantic import ValidationError

from .config import get_logger
from .decomposer import GroqJSONClient
from .models import RouteResult
from .prompts import render_routing_prompt

LOGGER = get_logger(__name__)


class QueryRouter:
    """Select the most suitable handling pipeline for a user query."""

    def __init__(self, groq_client: GroqJSONClient) -> None:
        self.groq_client = groq_client

    def route(self, query: str) -> RouteResult:
        if not query.strip():
            raise ValueError("query must not be empty.")

        payload = self.groq_client.invoke_json(render_routing_prompt(query.strip()))
        try:
            result = RouteResult.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(f"Invalid routing payload: {payload}") from exc

        LOGGER.info("Selected route '%s' for query: %s", result.route, query)
        return result
