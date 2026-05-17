"""Dense retrieval against Qdrant using BGE embeddings."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)
logger.info("[RETRIEVER] Module starting...")

from qdrant_client import QdrantClient
logger.info("[RETRIEVER] QdrantClient imported")
from tqdm.auto import tqdm
logger.info("[RETRIEVER] tqdm imported")

from .config import QueryEngineConfig, get_logger
logger.info("[RETRIEVER] Config imported")
from .embedding_model import get_embedding_model
logger.info("[RETRIEVER] Embedding model imported")
from .models import RetrievalQuery, RetrievedVerse
logger.info("[RETRIEVER] Models imported")

LOGGER = get_logger(__name__)
logger.info("[RETRIEVER] ✓ Module fully initialized")


class QdrantVerseRetriever:
    """Retrieve Bhagavad Gita verses for each semantic search query."""

    def __init__(self, config: QueryEngineConfig) -> None:
        self.config = config
        LOGGER.info("[RETRIEVER] Connecting to Qdrant at %s", config.qdrant_endpoint)
        self.client = QdrantClient(
            url=config.qdrant_endpoint,
            api_key=config.qdrant_api_key,
            timeout=config.qdrant_timeout_seconds,
        )
        LOGGER.info("[RETRIEVER] ✓ Qdrant client initialized successfully")

    def retrieve(
        self,
        retrieval_queries: list[RetrievalQuery],
        top_k: int | None = None,
    ) -> list[RetrievedVerse]:
        if not retrieval_queries:
            return []

        limit = top_k or self.config.qdrant_top_k_per_problem
        collected: list[RetrievedVerse] = []

        for item in tqdm(retrieval_queries, desc="Retrieving verses", leave=False):
            # Use singleton embedding model (loaded at startup)
            embedding_model = get_embedding_model()
            vector = embedding_model.encode(
                item.query,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).tolist()
            search_hits = self._search_hits(vector, limit)

            verses = [self._hit_to_model(hit, item) for hit in search_hits]
            LOGGER.info(
                "Retrieved verses for '%s': %s",
                item.query,
                [f"{verse.chapter}.{verse.verse} ({verse.retrieval_score:.4f})" for verse in verses],
            )
            collected.extend(verses)

        return collected

    def _search_hits(self, vector: list[float], limit: int) -> list[Any]:
        if hasattr(self.client, "search"):
            return self.client.search(
                collection_name=self.config.qdrant_collection_name,
                query_vector=vector,
                limit=limit,
                with_payload=True,
            )

        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.config.qdrant_collection_name,
                query=vector,
                limit=limit,
                with_payload=True,
            )
            points = getattr(response, "points", None)
            if points is None:
                raise RuntimeError("Qdrant query_points response did not include points.")
            return list(points)

        raise RuntimeError("Unsupported Qdrant client: neither 'search' nor 'query_points' is available.")

    def _hit_to_model(self, hit: Any, item: RetrievalQuery) -> RetrievedVerse:
        payload = dict(hit.payload or {})
        return RetrievedVerse(
            id=str(payload.get("id", "")) or None,
            chapter=int(payload.get("chapter", 0)),
            verse=int(payload.get("verse", 0)),
            speaker=str(payload.get("speaker", "")),
            shloka=str(payload.get("shloka", "")),
            translation=str(payload.get("translation", "")),
            interpretation=str(payload.get("interpretation", "")),
            topics=payload.get("topics", []),
            emotion_tags=payload.get("emotion_tags", []),
            summary=str(payload.get("summary", "")),
            retrieval_text=str(payload.get("retrieval_text", "")),
            score=float(hit.score),
            retrieval_score=float(hit.score),
            matched_problems=[item.problem],
            matched_emotions=[item.emotion],
            matched_queries=[item.query],
        )
