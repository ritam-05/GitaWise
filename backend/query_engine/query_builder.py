"""Build optimized dense retrieval queries from problems and emotions."""

from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError

from .config import get_logger
from .decomposer import GroqJSONClient
from .models import EmotionResult, RetrievalQuery
from .prompts import render_query_builder_prompt

LOGGER = get_logger(__name__)


class QueryEnvelope(BaseModel):
    """Expected LLM payload for retrieval query construction."""

    queries: list[RetrievalQuery] = Field(default_factory=list)


class RetrievalQueryBuilder:
    """Convert problem-emotion pairs into semantic retrieval queries."""

    def __init__(self, groq_client: GroqJSONClient) -> None:
        self.groq_client = groq_client

    def build(self, emotion_results: list[EmotionResult]) -> list[RetrievalQuery]:
        if not emotion_results:
            return []

        items = [
            {"problem": item.problem, "emotion": item.emotion}
            for item in emotion_results
        ]
        payload = self.groq_client.invoke_json(render_query_builder_prompt(items))
        try:
            envelope = QueryEnvelope.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(f"Invalid retrieval-query payload: {payload}") from exc

        result_map = {
            (item.problem.strip(), item.emotion): item
            for item in envelope.queries
            if item.query.strip()
        }

        ordered_queries: list[RetrievalQuery] = []
        for item in emotion_results:
            key = (item.problem.strip(), item.emotion)
            built_query = result_map.get(key)
            if built_query is None:
                fallback_text = item.problem if item.emotion == "none" else f"{item.problem} {item.emotion} duty clarity wisdom"
                built_query = RetrievalQuery(
                    problem=item.problem,
                    emotion=item.emotion,
                    query=fallback_text.strip(),
                )
            ordered_queries.append(built_query)

        LOGGER.info(
            "Generated retrieval queries: %s",
            [
                {"problem": item.problem, "emotion": item.emotion, "query": item.query}
                for item in ordered_queries
            ],
        )
        return ordered_queries
