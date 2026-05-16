"""Prepare citation-ready context objects from reranked verses."""

from __future__ import annotations

from .config import get_logger
from .models import RetrievedVerse

LOGGER = get_logger(__name__)


class ContextBuilder:
    """Build the final structured context passed to the generation layer."""

    def build(self, reranked_verses: list[RetrievedVerse], top_k: int) -> list[RetrievedVerse]:
        contexts = []
        for verse in reranked_verses[:top_k]:
            context = verse.model_copy(deep=True)
            context.score = round(float(context.score), 4)
            if context.retrieval_score is not None:
                context.retrieval_score = round(float(context.retrieval_score), 4)
            if context.rerank_score is not None:
                context.rerank_score = round(float(context.rerank_score), 4)
            contexts.append(context)

        LOGGER.info(
            "Built final context set: %s",
            [f"{item.chapter}.{item.verse} ({item.score:.4f})" for item in contexts],
        )
        return contexts
