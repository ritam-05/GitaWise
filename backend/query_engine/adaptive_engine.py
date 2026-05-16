"""Route-aware orchestration for the adaptive GitaWise backend."""

from __future__ import annotations

from .config import QueryEngineConfig, get_logger, load_query_engine_config
from .engine import GitaQueryEngine
from .generator import DirectResponseGenerator
from .models import AdaptiveAnswer, EmotionResult, Problem, RetrievalQuery, RetrievedVerse


class AdaptiveGitaEngine:
    """Hybrid orchestration layer that routes queries before any RAG work begins."""

    def __init__(self, config: QueryEngineConfig | None = None) -> None:
        self.config = config or load_query_engine_config()
        self.logger = get_logger(self.__class__.__name__, self.config.log_level)
        self.query_engine = GitaQueryEngine(self.config)
        self.direct_generator = DirectResponseGenerator(self.query_engine.groq_client)

    def answer(self, user_query: str) -> AdaptiveAnswer:
        if not user_query.strip():
            raise ValueError("user_query must not be empty.")

        route_result = self.query_engine.route_query(user_query)
        route = route_result.route
        self.logger.info("Adaptive route selected: %s", route)

        if route == "generic_chat":
            return self.direct_generator.generate(
                user_query=user_query,
                route=route,
                fallback_note="This query should be handled as direct conversation without Bhagavad Gita retrieval.",
            )

        if route == "gita_rag":
            problems = [Problem(problem=user_query.strip())]
            emotions = [EmotionResult(problem=user_query.strip(), emotion="none")]
            retrieval_queries = [
                RetrievalQuery(
                    problem=user_query.strip(),
                    emotion="none",
                    query=f"{user_query.strip()} Bhagavad Gita verse interpretation Krishna Arjuna",
                )
            ]
            return self._run_grounded_route(
                user_query=user_query,
                route=route,
                problems=problems,
                emotions=emotions,
                retrieval_queries=retrieval_queries,
            )

        problems = self.query_engine.decompose_query(user_query)
        emotions = self.query_engine.detect_emotions(problems)
        retrieval_queries = self.query_engine.build_queries(emotions)
        return self._run_grounded_route(
            user_query=user_query,
            route=route,
            problems=problems,
            emotions=emotions,
            retrieval_queries=retrieval_queries,
        )

    def _run_grounded_route(
        self,
        user_query: str,
        route: str,
        problems: list[Problem],
        emotions: list[EmotionResult],
        retrieval_queries: list[RetrievalQuery],
    ) -> AdaptiveAnswer:
        warnings: list[str] = []

        try:
            retrieved_verses = self.query_engine.retrieve(retrieval_queries)
        except Exception as exc:
            self.logger.exception("Route-aware retrieval failed.")
            warnings.append(f"retrieval_failed: {exc}")
            return self.direct_generator.generate(
                user_query=user_query,
                route=route,
                warnings=warnings,
                fallback_note="Bhagavad Gita retrieval could not be completed, so answer directly and cautiously.",
            )

        strong_verses = self._filter_by_confidence(retrieved_verses)
        if not strong_verses:
            warnings.append(
                f"low_retrieval_confidence: no verse met threshold {self.config.retrieval_confidence_threshold:.2f}"
            )
            return self.direct_generator.generate(
                user_query=user_query,
                route=route,
                warnings=warnings,
                fallback_note="Retrieved verse relevance was too weak for trustworthy grounding. Respond without fake citations.",
            )

        reranked_verses = self.query_engine.rerank(user_query, strong_verses)
        contexts = self.query_engine.build_context(reranked_verses)
        if not contexts:
            warnings.append("no_grounded_contexts")
            return self.direct_generator.generate(
                user_query=user_query,
                route=route,
                warnings=warnings,
                fallback_note="No reliable grounded contexts were available after retrieval. Respond directly and cautiously.",
            )

        generated = self.query_engine.generator.generate(
            user_query=user_query,
            problems=problems,
            emotions=emotions,
            contexts=contexts,
            warnings=warnings,
        )
        return AdaptiveAnswer(
            original_query=user_query,
            route=route,
            answer=generated.answer,
            cited_verses=generated.cited_verses,
            contexts=generated.contexts,
            warnings=generated.warnings,
            used_rag=True,
        )

    def _filter_by_confidence(self, verses: list[RetrievedVerse]) -> list[RetrievedVerse]:
        threshold = self.config.retrieval_confidence_threshold
        filtered = [
            verse for verse in verses if (verse.retrieval_score or 0.0) >= threshold
        ]
        self.logger.info(
            "Retained %s/%s verses above retrieval confidence threshold %.2f",
            len(filtered),
            len(verses),
            threshold,
        )
        return filtered
