"""End-to-end orchestration for emotion-aware Bhagavad Gita retrieval."""

from __future__ import annotations

import logging
logger = logging.getLogger(__name__)
logger.info("[ENGINE_MODULE] Starting engine module imports...")

from .config import QueryEngineConfig, get_logger, load_query_engine_config
logger.info("[ENGINE_MODULE] Config imported")
from .combined_analyzer import CombinedAnalyzer
logger.info("[ENGINE_MODULE] CombinedAnalyzer imported")
from .context_builder import ContextBuilder
logger.info("[ENGINE_MODULE] ContextBuilder imported")
from .decomposer import GroqJSONClient
logger.info("[ENGINE_MODULE] GroqJSONClient imported")
from .emotion_normalizer import EmotionNormalizer
logger.info("[ENGINE_MODULE] EmotionNormalizer imported")
from .generator import GroundedResponseGenerator
logger.info("[ENGINE_MODULE] GroundedResponseGenerator imported")
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
logger.info("[ENGINE_MODULE] Models imported")
from .query_builder import RetrievalQueryBuilder
logger.info("[ENGINE_MODULE] RetrievalQueryBuilder imported")
from .reranker import GlobalVerseReranker
logger.info("[ENGINE_MODULE] GlobalVerseReranker imported")
from .retriever import QdrantVerseRetriever
logger.info("[ENGINE_MODULE] QdrantVerseRetriever imported (WATCH: this may hang on Qdrant connect)")
from .router import QueryRouter
logger.info("[ENGINE_MODULE] QueryRouter imported")

logger.info("[ENGINE_MODULE] ✓ All engine module imports completed")


class GitaQueryEngine:
    """Emotion-aware adaptive retrieval engine for Bhagavad Gita grounding."""

    def __init__(self, config: QueryEngineConfig | None = None) -> None:
        self.config = config or load_query_engine_config()
        self.logger = get_logger(self.__class__.__name__, self.config.log_level)
        self.logger.info("[ENGINE] ✓ Initializing GitaQueryEngine...")
        self.logger.info("[ENGINE] Creating Groq client...")
        self.groq_client = GroqJSONClient(self.config)
        self.logger.info("[ENGINE] ✓ Groq client created")
        self.logger.info("[ENGINE] Creating router...")
        self.router = QueryRouter(self.groq_client)
        self.logger.info("[ENGINE] ✓ Router created")
        self.logger.info("[ENGINE] Creating combined analyzer...")
        self.combined_analyzer = CombinedAnalyzer(self.groq_client, self.config)
        self.logger.info("[ENGINE] ✓ Combined analyzer created")
        self.logger.info("[ENGINE] Creating emotion normalizer...")
        self.emotion_normalizer = EmotionNormalizer(self.groq_client)
        self.logger.info("[ENGINE] ✓ Emotion normalizer created")
        self.logger.info("[ENGINE] Creating generator...")
        self.generator = GroundedResponseGenerator(self.groq_client)
        self.logger.info("[ENGINE] ✓ Generator created")
        self.logger.info("[ENGINE] Creating query builder...")
        self.query_builder = RetrievalQueryBuilder(self.groq_client)
        self.logger.info("[ENGINE] ✓ Query builder created")
        self.logger.info("[ENGINE] Creating retriever...")
        self.retriever = QdrantVerseRetriever(self.config)
        self.logger.info("[ENGINE] ✓ Retriever created")
        self.logger.info("[ENGINE] Creating reranker...")
        self.reranker = GlobalVerseReranker(self.config)
        self.logger.info("[ENGINE] ✓ Reranker created")
        self.logger.info("[ENGINE] Creating context builder...")
        self.context_builder = ContextBuilder()
        self.logger.info("[ENGINE] ✓ Context builder created")
        self.logger.info("[ENGINE] ✓✓✓ GitaQueryEngine fully initialized!")

    def decompose_query(self, user_query: str) -> list[Problem]:
        """Deprecated: use analyze_query instead."""
        problems, _ = self.combined_analyzer.analyze(user_query)
        return problems

    def route_query(self, user_query: str) -> RouteResult:
        return self.router.route(user_query)

    def detect_emotions(self, problems: list[Problem]) -> list[EmotionResult]:
        """Deprecated: use analyze_query instead."""
        raise NotImplementedError(
            "detect_emotions is deprecated. Use analyze_query() for combined decomposition+emotion detection."
        )

    def analyze_query(self, user_query: str) -> tuple[list[Problem], list[EmotionResult]]:
        """
        Unified query analysis: decompose and detect emotions in ONE efficient LLM call.

        This replaces the separate decompose_query + detect_emotions pipeline.
        """
        return self.combined_analyzer.analyze(user_query)

    def normalize_emotion(self, emotion: str) -> EmotionNormalizationResult:
        return self.emotion_normalizer.normalize(emotion)

    def build_queries(self, emotion_results: list[EmotionResult]) -> list[RetrievalQuery]:
        return self.query_builder.build(emotion_results)

    def retrieve(self, retrieval_queries: list[RetrievalQuery]) -> list[RetrievedVerse]:
        return self.retriever.retrieve(retrieval_queries, top_k=self.config.qdrant_top_k_per_problem)

    def rerank(self, original_query: str, candidates: list[RetrievedVerse]) -> list[RetrievedVerse]:
        return self.reranker.rerank(original_query, candidates, top_k=self.config.final_top_k)

    def build_context(self, reranked_verses: list[RetrievedVerse]) -> list[RetrievedVerse]:
        return self.context_builder.build(reranked_verses, top_k=self.config.generation_context_top_k)

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

        # UNIFIED ANALYSIS: decompose + detect emotions in ONE LLM call
        try:
            problems, emotions = self.analyze_query(user_query)
        except Exception as exc:
            self.logger.exception("Combined analysis failed.")
            warnings.append(f"combined_analysis_failed: {exc}")
            # Fallback: treat as generic query
            problems = [Problem(problem="user query")]
            emotions = [EmotionResult(problem="user query", emotion="none")]

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
