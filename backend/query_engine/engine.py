"""End-to-end orchestration for emotion-aware Bhagavad Gita retrieval."""

from __future__ import annotations

from .config import QueryEngineConfig, get_logger, load_query_engine_config
from .context_builder import ContextBuilder
from .decomposer import GroqJSONClient, QueryDecomposer
from .emotion_detector import EmotionDetector
from .emotion_normalizer import EmotionNormalizer
from .generator import GroundedResponseGenerator
from .models import (
    EmotionNormalizationResult,
    EmotionResult,
    EngineResponse,
    GeneratedAnswer,
    Problem,
    RouteResult,
    RetrievalQuery,
    RetrievedVerse,
)
from .query_builder import RetrievalQueryBuilder
from .reranker import GlobalVerseReranker
from .retriever import QdrantVerseRetriever
from .router import QueryRouter


class GitaQueryEngine:
    """Emotion-aware adaptive retrieval engine for Bhagavad Gita grounding."""

    def __init__(self, config: QueryEngineConfig | None = None) -> None:
        self.config = config or load_query_engine_config()
        self.logger = get_logger(self.__class__.__name__, self.config.log_level)
        self.groq_client = GroqJSONClient(self.config)
        self.router = QueryRouter(self.groq_client)
        self.decomposer = QueryDecomposer(self.groq_client)
        self.emotion_detector = EmotionDetector(self.groq_client)
        self.emotion_normalizer = EmotionNormalizer(self.groq_client)
        self.generator = GroundedResponseGenerator(self.groq_client)
        self.query_builder = RetrievalQueryBuilder(self.groq_client)
        self.retriever = QdrantVerseRetriever(self.config)
        self.reranker = GlobalVerseReranker(self.config)
        self.context_builder = ContextBuilder()

    def decompose_query(self, user_query: str) -> list[Problem]:
        return self.decomposer.decompose(user_query)

    def route_query(self, user_query: str) -> RouteResult:
        return self.router.route(user_query)

    def detect_emotions(self, problems: list[Problem]) -> list[EmotionResult]:
        return self.emotion_detector.detect(problems)

    def normalize_emotion(self, emotion: str) -> EmotionNormalizationResult:
        return self.emotion_normalizer.normalize(emotion)

    def build_queries(self, emotion_results: list[EmotionResult]) -> list[RetrievalQuery]:
        return self.query_builder.build(emotion_results)

    def retrieve(self, retrieval_queries: list[RetrievalQuery]) -> list[RetrievedVerse]:
        return self.retriever.retrieve(retrieval_queries, top_k=self.config.qdrant_top_k_per_problem)

    def rerank(self, original_query: str, candidates: list[RetrievedVerse]) -> list[RetrievedVerse]:
        return self.reranker.rerank(original_query, candidates, top_k=self.config.final_top_k)

    def build_context(self, reranked_verses: list[RetrievedVerse]) -> list[RetrievedVerse]:
        return self.context_builder.build(reranked_verses, top_k=self.config.final_top_k)

    def generate_answer(self, retrieval_result: EngineResponse) -> GeneratedAnswer:
        return self.generator.generate(
            user_query=retrieval_result.original_query,
            problems=retrieval_result.problems,
            emotions=retrieval_result.emotions,
            contexts=retrieval_result.contexts,
            warnings=retrieval_result.warnings,
        )

    def run(self, user_query: str) -> EngineResponse:
        if not user_query.strip():
            raise ValueError("user_query must not be empty.")

        self.logger.info("Starting query-engine pipeline for query: %s", user_query)
        warnings: list[str] = []
        problems = self.decompose_query(user_query)
        emotions = self.detect_emotions(problems)
        retrieval_queries = self.build_queries(emotions)
        try:
            retrieved_verses = self.retrieve(retrieval_queries)
        except Exception as exc:
            self.logger.exception("Verse retrieval failed.")
            warnings.append(f"retrieval_failed: {exc}")
            retrieved_verses = []

        try:
            reranked_verses = self.rerank(user_query, retrieved_verses)
        except Exception as exc:
            self.logger.exception("Verse reranking failed.")
            warnings.append(f"reranking_failed: {exc}")
            reranked_verses = sorted(
                retrieved_verses,
                key=lambda item: item.retrieval_score or 0.0,
                reverse=True,
            )[: self.config.final_top_k]

        try:
            contexts = self.build_context(reranked_verses)
        except Exception as exc:
            self.logger.exception("Context building failed.")
            warnings.append(f"context_building_failed: {exc}")
            contexts = []

        response = EngineResponse(
            original_query=user_query,
            problems=problems,
            emotions=emotions,
            retrieval_queries=retrieval_queries,
            contexts=contexts,
            warnings=warnings,
        )
        self.logger.info("Query-engine pipeline completed with %s final contexts.", len(contexts))
        return response
