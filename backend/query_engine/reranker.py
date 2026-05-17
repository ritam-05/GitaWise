"""Global reranking for merged retrieval candidates."""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)
logger.info("[RERANKER] Module starting...")

import torch
logger.info("[RERANKER] Torch imported")

from .config import QueryEngineConfig, get_logger
logger.info("[RERANKER] Config imported")
from .reranker_model import get_reranker_model, get_reranker_tokenizer
logger.info("[RERANKER] Reranker model functions imported (lazy)")
from .models import RetrievedVerse
logger.info("[RERANKER] Models imported")

LOGGER = get_logger(__name__)
logger.info("[RERANKER] ✓ Module fully initialized")


class GlobalVerseReranker:
    """Rerank merged verse candidates with a cross-encoder reranker."""

    def __init__(self, config: QueryEngineConfig) -> None:
        self.config = config
        # Models are loaded at app startup, not per-request
        # Get device from singleton (same as used during initialization)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def rerank(
        self,
        original_query: str,
        candidates: list[RetrievedVerse],
        top_k: int | None = None,
    ) -> list[RetrievedVerse]:
        if not candidates:
            return []

        limit = top_k or self.config.final_top_k

        # Step 1: deduplicate merged candidates
        deduplicated = self._deduplicate(candidates)

        # Step 2: fast embedding-based filter using retrieval_score produced by Qdrant
        # Keep top-N candidates by retrieval_score to reduce cross-encoder cost.
        filtered_by_embedding = sorted(
            deduplicated, key=lambda item: (item.retrieval_score or 0.0), reverse=True
        )[: self.config.reranker_embed_filter_top_k]

        # Step 3: build concise rerank query and score pairs using cross-encoder
        rerank_query = self._build_rerank_query(original_query, filtered_by_embedding)
        passages = [self._candidate_text(item) for item in filtered_by_embedding]
        scores = self._score_pairs(rerank_query, passages)

        # Attach scores back to the filtered items
        for item, score in zip(filtered_by_embedding, scores):
            item.rerank_score = score
            # Combine signals: use rerank score as primary but preserve retrieval_score
            item.score = score

        # Final ranking: prefer rerank score, break ties with retrieval_score
        ranked = sorted(
            filtered_by_embedding,
            key=lambda item: (item.score, item.retrieval_score or 0.0),
            reverse=True,
        )[:limit]

        LOGGER.info(
            "Hybrid reranked verses (top %s of %s): %s",
            limit,
            len(filtered_by_embedding),
            [f"{item.chapter}.{item.verse} ({item.score:.4f})" for item in ranked],
        )
        return ranked

    def _score_pairs(self, query: str, passages: list[str]) -> list[float]:
        """Score query-passage pairs using the singleton reranker model."""
        model = get_reranker_model()
        tokenizer = get_reranker_tokenizer()
        all_scores: list[float] = []

        for start in range(0, len(passages), self.config.reranker_batch_size):
            batch_passages = passages[start : start + self.config.reranker_batch_size]
            pairs = [[query, passage] for passage in batch_passages]
            inputs = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            with torch.no_grad():
                logits = model(**inputs, return_dict=True).logits.view(-1).float()

            all_scores.extend(self._normalize_scores(logits.cpu().tolist()))

        return all_scores

    @staticmethod
    def _normalize_scores(raw_scores: list[float]) -> list[float]:
        """Sigmoid normalization for cross-encoder scores."""
        return [1.0 / (1.0 + math.exp(-float(score))) for score in raw_scores]

    def _build_rerank_query(self, original_query: str, candidates: list[RetrievedVerse]) -> str:
        concerns: list[str] = []
        for item in candidates:
            for problem, emotion in zip(item.matched_problems, item.matched_emotions):
                concern = f"{problem} [{emotion}]"
                if concern not in concerns:
                    concerns.append(concern)

        if not concerns:
            return original_query.strip()
        return (
            f"{original_query.strip()}\n"
            f"Philosophical concerns: {'; '.join(concerns)}"
        )

    @staticmethod
    def _candidate_text(candidate: RetrievedVerse) -> str:
        if candidate.retrieval_text.strip():
            return candidate.retrieval_text
        return "\n".join(
            part
            for part in [
                f"Chapter {candidate.chapter} Verse {candidate.verse}",
                candidate.translation,
                candidate.interpretation,
                candidate.summary,
                ", ".join(candidate.topics),
                ", ".join(candidate.emotion_tags),
            ]
            if part.strip()
        )

    @staticmethod
    def _deduplicate(candidates: list[RetrievedVerse]) -> list[RetrievedVerse]:
        merged: dict[tuple[int, int], RetrievedVerse] = {}
        for item in candidates:
            key = item.verse_key
            existing = merged.get(key)
            if existing is None:
                merged[key] = item.model_copy(deep=True)
                continue

            retrieval_scores = [
                score
                for score in [existing.retrieval_score, item.retrieval_score]
                if score is not None
            ]
            existing.retrieval_score = max(retrieval_scores) if retrieval_scores else None
            for problem in item.matched_problems:
                if problem not in existing.matched_problems:
                    existing.matched_problems.append(problem)
            for emotion in item.matched_emotions:
                if emotion not in existing.matched_emotions:
                    existing.matched_emotions.append(emotion)
            for query in item.matched_queries:
                if query not in existing.matched_queries:
                    existing.matched_queries.append(query)

        return list(merged.values())
