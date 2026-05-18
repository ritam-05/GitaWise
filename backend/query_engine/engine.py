"""End-to-end orchestration for emotion-aware Bhagavad Gita retrieval."""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)
logger.info("[ENGINE_MODULE] Starting engine module imports...")

from .combined_analyzer import CombinedAnalyzer
from .config import QueryEngineConfig, get_logger, load_query_engine_config
from .context_builder import ContextBuilder
from .decomposer import GroqJSONClient
from .emotion_normalizer import EmotionNormalizer
from .generator import GroundedResponseGenerator
from .lightweight_router import LightweightRouter
from .models import (
    EmotionNormalizationResult,
    EmotionResult,
    EngineResponse,
    GeneratedAnswer,
    Problem,
    RetrievalQuery,
    RetrievedVerse,
    RouteResult,
)
from .query_builder import RetrievalQueryBuilder
from .reranker import GlobalVerseReranker
from .retriever import QdrantVerseRetriever

logger.info("[ENGINE_MODULE] Core query-engine modules imported")

try:
    from backend.cache import CacheManager, SessionCache

    CACHE_AVAILABLE = True
    logger.info("[ENGINE_MODULE] Cache layer imported")
except ImportError:
    CACHE_AVAILABLE = False
    CacheManager = object
    SessionCache = object
    logger.info("[ENGINE_MODULE] Cache layer not available (proceeding without caching)")


class GitaQueryEngine:
    """Emotion-aware adaptive retrieval engine for Bhagavad Gita grounding."""

    def __init__(self, config: QueryEngineConfig | None = None, cache_manager: Optional[CacheManager] = None) -> None:
        self.config = config or load_query_engine_config()
        self.logger = get_logger(self.__class__.__name__, self.config.log_level)
        self.logger.info("[ENGINE] Initializing GitaQueryEngine...")

        self.cache_manager = cache_manager
        self.session_cache: Optional[SessionCache] = None
        if CACHE_AVAILABLE and self.cache_manager is None:
            self.cache_manager = CacheManager()
            self.session_cache = SessionCache(
                self.cache_manager,
                session_ttl_seconds=self.config.session_ttl_seconds,
                session_max_stored_turns=self.config.session_max_stored_turns,
                session_context_turns=self.config.session_context_turns,
                session_cleanup_interval_seconds=self.config.session_cleanup_interval_seconds,
            )
            self.logger.info("[ENGINE] Cache initialized (session-aware, multi-layer)")
        elif CACHE_AVAILABLE and self.cache_manager:
            self.session_cache = SessionCache(
                self.cache_manager,
                session_ttl_seconds=self.config.session_ttl_seconds,
                session_max_stored_turns=self.config.session_max_stored_turns,
                session_context_turns=self.config.session_context_turns,
                session_cleanup_interval_seconds=self.config.session_cleanup_interval_seconds,
            )
            self.logger.info("[ENGINE] Using provided cache manager")
        else:
            self.logger.warning("[ENGINE] Running without caching (cache_available=%s)", CACHE_AVAILABLE)

        self.logger.info("[ENGINE] Creating Groq JSON client...")
        self.groq_client = GroqJSONClient(self.config)
        self.logger.info("[ENGINE] Groq JSON client created")

        self.logger.info("[ENGINE] Creating lightweight router...")
        self.router = LightweightRouter()
        self.logger.info("[ENGINE] Lightweight router created")

        self.logger.info("[ENGINE] Creating combined analyzer...")
        self.combined_analyzer = CombinedAnalyzer(self.groq_client, self.config)
        self.logger.info("[ENGINE] Combined analyzer created")

        self.logger.info("[ENGINE] Creating emotion normalizer...")
        self.emotion_normalizer = EmotionNormalizer(self.groq_client)
        self.logger.info("[ENGINE] Emotion normalizer created")

        self.logger.info("[ENGINE] Creating response generator...")
        self.generator = GroundedResponseGenerator(self.groq_client)
        self.logger.info("[ENGINE] Response generator created")

        self.logger.info("[ENGINE] Creating query builder...")
        self.query_builder = RetrievalQueryBuilder()
        self.logger.info("[ENGINE] Query builder created")

        self.logger.info("[ENGINE] Creating retriever...")
        self.retriever = QdrantVerseRetriever(self.config)
        self.logger.info("[ENGINE] Retriever created")

        self.logger.info("[ENGINE] Creating reranker...")
        self.reranker = GlobalVerseReranker(self.config)
        self.logger.info("[ENGINE] Reranker created")

        self.logger.info("[ENGINE] Creating context builder...")
        self.context_builder = ContextBuilder()
        self.logger.info("[ENGINE] Context builder created")
        self.logger.info("[ENGINE] GitaQueryEngine fully initialized")

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
        """Unified query analysis: decompose and detect emotions in one LLM call."""
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

        try:
            problems, emotions = self.analyze_query(user_query)
        except Exception as exc:
            self.logger.exception("Combined analysis failed.")
            warnings.append(f"combined_analysis_failed: {exc}")
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

    def run_with_session(self, user_query: str, session_id: str) -> EngineResponse:
        """Session-aware query pipeline with intelligent caching."""
        if not user_query.strip():
            raise ValueError("user_query must not be empty.")

        if not self.session_cache:
            self.logger.warning("[ENGINE] Session cache not available, falling back to regular run()")
            return self.run(user_query)

        self.logger.info("[ENGINE_SESSION] Processing query with session_id=%s", session_id)
        warnings: list[str] = []

        try:
            cached_decomp = self.session_cache.get_cached_decomposition(user_query)
            cached_emotions_list = self.session_cache.get_cached_emotions(user_query)

            if cached_decomp and cached_emotions_list:
                self.logger.info("[ENGINE_SESSION] Cache hit: decomposition + emotions")
                problems = [Problem(problem=p["problem"]) for p in cached_decomp]
                emotions = [
                    EmotionResult(problem=p, emotion=e)
                    for p, e in zip([p["problem"] for p in cached_decomp], cached_emotions_list)
                ]
                warnings.append("used_cache:decomposition_emotions")
            else:
                self.logger.info("[ENGINE_SESSION] Cache miss: running fresh decomposition + emotion analysis")
                problems, emotions = self.analyze_query(user_query)
                self.session_cache.cache_decomposition(
                    session_id,
                    user_query,
                    [{"problem": p.problem} for p in problems],
                )
                self.session_cache.cache_emotions(
                    session_id,
                    user_query,
                    [e.emotion for e in emotions],
                )
        except Exception as exc:
            self.logger.exception("[ENGINE_SESSION] Analysis failed: %s", exc)
            warnings.append(f"analysis_failed: {exc}")
            problems = [Problem(problem="user query")]
            emotions = [EmotionResult(problem="user query", emotion="none")]

        retrieval_queries = self.build_queries(emotions)

        try:
            cached_verses = None
            for ret_q in retrieval_queries:
                cached = self.session_cache.get_cached_retrieval(ret_q.query)
                if cached:
                    cached_verses = cached
                    self.logger.info("[ENGINE_SESSION] Cache hit: %d retrieved verses", len(cached))
                    warnings.append("used_cache:retrieval")
                    break

            if cached_verses:
                retrieved_verses = [RetrievedVerse(**v) if isinstance(v, dict) else v for v in cached_verses]
            else:
                self.logger.info("[ENGINE_SESSION] Cache miss: running fresh retrieval")
                retrieved_verses = self.retrieve(retrieval_queries)
                if retrieved_verses:
                    self.session_cache.cache_retrieval(
                        session_id,
                        retrieval_queries[0].query if retrieval_queries else "unknown",
                        [v.model_dump() if hasattr(v, "model_dump") else v for v in retrieved_verses],
                    )
        except Exception as exc:
            self.logger.exception("[ENGINE_SESSION] Retrieval failed: %s", exc)
            warnings.append(f"retrieval_failed: {exc}")
            retrieved_verses = []

        try:
            reranked_verses = self.rerank(user_query, retrieved_verses)
        except Exception as exc:
            self.logger.exception("[ENGINE_SESSION] Reranking failed: %s", exc)
            warnings.append(f"reranking_failed: {exc}")
            reranked_verses = sorted(
                retrieved_verses,
                key=lambda item: item.retrieval_score or 0.0,
                reverse=True,
            )[: self.config.final_top_k]

        try:
            contexts = self.build_context(reranked_verses)
        except Exception as exc:
            self.logger.exception("[ENGINE_SESSION] Context building failed: %s", exc)
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
        self.logger.info(
            "[ENGINE_SESSION] Pipeline completed with %d final contexts (session_id=%s)",
            len(contexts),
            session_id,
        )
        return response
