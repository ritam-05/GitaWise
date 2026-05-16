"""Global reranking for merged retrieval candidates."""

from __future__ import annotations

import math

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import QueryEngineConfig, get_logger
from .models import RetrievedVerse

LOGGER = get_logger(__name__)


class GlobalVerseReranker:
    """Rerank merged verse candidates with a cross-encoder reranker."""

    def __init__(self, config: QueryEngineConfig) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(config.reranker_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.reranker_model_name
        )
        self.model.to(self.device)
        self.model.eval()

    def rerank(
        self,
        original_query: str,
        candidates: list[RetrievedVerse],
        top_k: int | None = None,
    ) -> list[RetrievedVerse]:
        if not candidates:
            return []

        limit = top_k or self.config.final_top_k
        deduplicated = self._deduplicate(candidates)
        rerank_query = self._build_rerank_query(original_query, deduplicated)
        passages = [self._candidate_text(item) for item in deduplicated]
        scores = self._score_pairs(rerank_query, passages)

        for item, score in zip(deduplicated, scores):
            item.rerank_score = score
            item.score = score

        ranked = sorted(
            deduplicated,
            key=lambda item: (item.score, item.retrieval_score or 0.0),
            reverse=True,
        )[:limit]

        LOGGER.info(
            "Reranked verses: %s",
            [f"{item.chapter}.{item.verse} ({item.score:.4f})" for item in ranked],
        )
        return ranked

    def _score_pairs(self, query: str, passages: list[str]) -> list[float]:
        all_scores: list[float] = []

        for start in range(0, len(passages), self.config.reranker_batch_size):
            batch_passages = passages[start : start + self.config.reranker_batch_size]
            pairs = [[query, passage] for passage in batch_passages]
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs, return_dict=True).logits.view(-1).float()

            all_scores.extend(self._normalize_scores(logits.cpu().tolist()))

        return all_scores

    @staticmethod
    def _normalize_scores(raw_scores: list[float]) -> list[float]:
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
