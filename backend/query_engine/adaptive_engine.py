"""Route-aware orchestration for the adaptive GitaWise backend."""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)
logger.info("[ADAPTIVE] Module starting...")

from .config import QueryEngineConfig, get_logger, load_query_engine_config
logger.info("[ADAPTIVE] Config imported")
from .engine import GitaQueryEngine
logger.info("[ADAPTIVE] Engine imported")
from .generator import DirectResponseGenerator
logger.info("[ADAPTIVE] Generator imported")
from .models import AdaptiveAnswer, EmotionResult, Problem, RetrievalQuery, RetrievedVerse
logger.info("[ADAPTIVE] Models imported")

from .context_resolver import ContextResolver
logger.info("[ADAPTIVE] ContextResolver imported")
from .dialogue_state import DialogueState
from .continuation_detector import ContinuationDetector
from .contextual_rewriter import ContextualRewriter
from .state_transition_manager import StateTransitionManager
logger.info("[ADAPTIVE] Dialogue helpers imported")

# Import cache (optional)
try:
    from backend.cache import CacheManager, SessionCache, ConversationTurn, SessionMemory
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    CacheManager = None
    SessionCache = None
    ConversationTurn = None
    SessionMemory = None


class AdaptiveGitaEngine:
    """Hybrid orchestration layer that routes queries before any RAG work begins."""

    def __init__(self, config: QueryEngineConfig | None = None, cache_manager: Optional[object] = None) -> None:
        self.config = config or load_query_engine_config()
        self.logger = get_logger(self.__class__.__name__, self.config.log_level)
        self.query_engine = GitaQueryEngine(self.config, cache_manager=cache_manager)
        self.direct_generator = DirectResponseGenerator(self.query_engine.groq_client)
        self.session_cache = self.query_engine.session_cache if hasattr(self.query_engine, 'session_cache') else None
        # Lightweight conversational resolver for follow-up resolution
        try:
            self.context_resolver = ContextResolver(self.query_engine.groq_client, self.config)
        except Exception:
            self.context_resolver = None
        # Continuation detector + contextual rewriter for continuation semantics
        try:
            self.continuation_detector = ContinuationDetector()
        except Exception:
            self.continuation_detector = None

        try:
            self.contextual_rewriter = ContextualRewriter(self.query_engine.groq_client, self.config)
        except Exception:
            self.contextual_rewriter = None
        try:
            self.transition_manager = StateTransitionManager()
        except Exception:
            self.transition_manager = None

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

        problems, emotions = self.query_engine.analyze_query(user_query)
        retrieval_queries = self.query_engine.build_queries(emotions)
        return self._run_grounded_route(
            user_query=user_query,
            route=route,
            problems=problems,
            emotions=emotions,
            retrieval_queries=retrieval_queries,
        )

    def answer_with_session(self, user_query: str, session_id: str) -> AdaptiveAnswer:
        """
        Answer with session-aware context chaining for multi-turn conversations.
        
        Each response builds on previous conversation turns.
        
        Args:
            user_query: Current user query
            session_id: Session identifier for context preservation
            
        Returns:
            AdaptiveAnswer with conversation-aware response
        """
        if not user_query.strip():
            raise ValueError("user_query must not be empty.")

        if not self.session_cache:
            self.logger.warning("[ADAPTIVE_SESSION] No session cache, falling back to stateless answer()")
            return self.answer(user_query)

        # Get or create session
        try:
            session = self.session_cache.get_or_create_session(session_id)
            conversation_history = self._format_conversation_history(session)
            self.logger.info("[ADAPTIVE_SESSION] Using session %s with %d prior turns", session_id, len(session.recent_turns))
        except Exception as exc:
            self.logger.warning("[ADAPTIVE_SESSION] Failed to load session: %s, falling back to stateless", exc)
            return self.answer(user_query)

        # Build structured dialogue state from the session and classify transition
        dialogue_state = DialogueState.from_session(session)
        resolved_query = user_query
        is_continuation = False

        try:
            classification = None
            if self.transition_manager:
                classification = self.transition_manager.classify_transition(user_query, dialogue_state, conversation_history)
            else:
                # fallback to previous continuation detector
                classification = {"transition_type": "continuation"} if (self.continuation_detector and self.continuation_detector.is_continuation(user_query, conversation_history)) else {"transition_type": "topic_shift"}

            ttype = classification.get("transition_type")
            # record transition type in dialogue state for downstream use
            try:
                dialogue_state.last_transition_type = ttype
                dialogue_state.conversation_state = ttype
            except Exception:
                pass

            if ttype == "emotion_shift":
                # Emotion escalation should override prior topic inheritance
                resolved_query = (self.contextual_rewriter.rewrite(user_query, dialogue_state, conversation_history, transition_type="emotion_shift")
                                  if self.contextual_rewriter else f"{user_query}")
                route = "emotion_guidance"
                self.logger.info("[ADAPTIVE_SESSION] Emotion shift detected; routing to %s", route)

            elif ttype == "topic_shift":
                # Topic shift: prefer explicit topic if provided
                topic = classification.get("topic")
                if topic:
                    # update dialogue state immediately to reflect new topic
                    dialogue_state.active_topic = topic
                # rewrite conservatively: if rewriter exists, allow it to craft a retrieval-friendly query
                if self.contextual_rewriter:
                    resolved_query = self.contextual_rewriter.rewrite(user_query, dialogue_state, conversation_history, transition_type="topic_shift")
                else:
                    resolved_query = user_query
                # allow router to select appropriate route (e.g., gita_rag)
                self.logger.info("[ADAPTIVE_SESSION] Topic shift detected; topic=%s", topic)

            else:
                # Continuation or fallback: continue prior topic when possible
                is_continuation = ttype == "continuation"
                if self.contextual_rewriter:
                    resolved_query = self.contextual_rewriter.rewrite(user_query, dialogue_state, conversation_history, transition_type=ttype)
                else:
                    base = (
                        dialogue_state.active_topic
                        or dialogue_state.last_resolved_query
                        or (session.recent_turns[-1].user_query if session.recent_turns else user_query)
                    )
                    resolved_query = f"Continue explaining Bhagavad Gita teachings about {base}"
                # inherit previous route to avoid aggressive rerouting
                route = dialogue_state.active_route or getattr(session, 'last_route', None)
                self.logger.info("[ADAPTIVE_SESSION] Continuation/fallback detected; resolved_query='%s' inherit_route=%s", resolved_query, route)

            # Persist resolved followup for record when applicable
            try:
                if resolved_query != user_query and hasattr(self.session_cache, 'record_resolved_followup'):
                    self.session_cache.record_resolved_followup(session.session_id, user_query, resolved_query)
            except Exception as exc:
                self.logger.warning("[ADAPTIVE_SESSION] Failed to record resolved followup: %s", exc)

        except Exception as exc:
            self.logger.warning("[ADAPTIVE_SESSION] Transition classification/rewriting failed: %s", exc)

        # Augment conversation history with compact dialogue state summary for generator
        try:
            state_summary = (
                f"DialogueState: topic={dialogue_state.active_topic}; emotions={','.join(dialogue_state.active_emotions or [])}; transition={getattr(dialogue_state, 'last_transition_type', None)}"
            )
            if conversation_history is None:
                conversation_history = []
            conversation_history = conversation_history + [{"role": "system", "content": state_summary}]
        except Exception:
            pass

        # Route query using the resolved query — preserve inherited route when available
        if is_continuation and 'route' in locals() and route:
            self.logger.info("[ADAPTIVE_SESSION] Using inherited route: %s (resolved_query='%s')", route, resolved_query)
        else:
            route_result = self.query_engine.route_query(resolved_query)
            route = route_result.route
            self.logger.info("[ADAPTIVE_SESSION] Route: %s (resolved_query='%s')", route, resolved_query)

        # Generate answer with conversation context
        if route == "generic_chat":
            answer = self.direct_generator.generate(
                user_query=resolved_query,
                route=route,
                fallback_note="This query should be handled as direct conversation without Bhagavad Gita retrieval.",
                conversation_history=conversation_history,
            )
        elif route == "gita_rag":
            problems = [Problem(problem=resolved_query.strip())]
            emotions = [EmotionResult(problem=resolved_query.strip(), emotion="none")]
            retrieval_queries = [
                RetrievalQuery(
                    problem=resolved_query.strip(),
                    emotion="none",
                    query=f"{resolved_query.strip()} Bhagavad Gita verse interpretation Krishna Arjuna",
                )
            ]
            answer = self._run_grounded_route(
                user_query=resolved_query,
                route=route,
                problems=problems,
                emotions=emotions,
                retrieval_queries=retrieval_queries,
                conversation_history=conversation_history,
            )
        else:
            problems, emotions = self.query_engine.analyze_query(resolved_query)
            retrieval_queries = self.query_engine.build_queries(emotions)
            answer = self._run_grounded_route(
                user_query=resolved_query,
                route=route,
                problems=problems,
                emotions=emotions,
                retrieval_queries=retrieval_queries,
                conversation_history=conversation_history,
            )

        # Track this turn in session and update structured dialogue state
        try:
            self._track_conversation_turn(
                session=session,
                user_query=resolved_query,
                route=route,
                problems=answer.contexts,
                emotions=[],
                response=answer.answer,
            )
            self.session_cache.save_session(session)
            self.logger.debug("[ADAPTIVE_SESSION] Tracked turn in session %s", session_id)
        except Exception as exc:
            self.logger.warning("[ADAPTIVE_SESSION] Failed to track turn: %s", exc)

        # Update and persist structured dialogue state for continuation semantics
        try:
            state = DialogueState.from_session(session)
            state.active_route = route
            state.last_resolved_query = resolved_query
            # Try to infer topic from grounded contexts
            topic = None
            try:
                if getattr(answer, 'contexts', None):
                    first = answer.contexts[0]
                    if isinstance(first, dict):
                        topic = first.get('theme') or first.get('title') or first.get('text')
                if not topic and isinstance(resolved_query, str):
                    import re

                    m = re.search(r"about (.+)$", resolved_query, re.IGNORECASE)
                    if m:
                        topic = m.group(1).strip()
            except Exception:
                topic = None

            if topic:
                state.active_topic = topic
            state.active_emotions = getattr(session, 'detected_emotions', []) or []
            state.apply_to_session(session)
            self.session_cache.save_session(session)
            self.logger.debug("[ADAPTIVE_SESSION] Updated dialogue state for session %s", session_id)
        except Exception as exc:
            self.logger.warning("[ADAPTIVE_SESSION] Failed to update dialogue state: %s", exc)

        return answer

    def _run_grounded_route(
        self,
        user_query: str,
        route: str,
        problems: list[Problem],
        emotions: list[EmotionResult],
        retrieval_queries: list[RetrievalQuery],
        conversation_history: list[dict[str, str]] | None = None,
    ) -> AdaptiveAnswer:
        """
        Run the grounded retrieval route with optional conversation context.
        
        Args:
            user_query: Current user query
            route: Selected route
            problems: Identified problems
            emotions: Detected emotions
            retrieval_queries: Queries for retrieval
            conversation_history: Optional previous conversation turns
        """
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
                conversation_history=conversation_history,
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
                conversation_history=conversation_history,
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
                conversation_history=conversation_history,
            )

        generated = self.query_engine.generator.generate(
            user_query=user_query,
            problems=problems,
            emotions=emotions,
            contexts=contexts,
            warnings=warnings,
            conversation_history=conversation_history,
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

    def _format_conversation_history(self, session: Optional[object]) -> list[dict[str, str]]:
        """
        Format session conversation turns into conversation history format.
        
        Uses config value for how many recent turns to include.
        
        Args:
            session: SessionMemory object (if available)
            
        Returns:
            List of {"role": "user"/"assistant", "content": "..."} dicts limited to session_context_turns
        """
        if not session or not hasattr(session, 'recent_turns'):
            return []

        history = []
        context_turns = self.config.session_context_turns
        
        try:
            # Get the last N turns based on config
            recent_turns = session.recent_turns[-context_turns:] if session.recent_turns else []
            
            for turn in recent_turns:
                # Add user query
                history.append({
                    "role": "user",
                    "content": turn.user_query
                })
                # Add assistant response
                history.append({
                    "role": "assistant",
                    "content": turn.response
                })
            
            self.logger.debug("[ADAPTIVE_SESSION] Formatted %d turns for context (max: %d)", len(recent_turns), context_turns)
        except Exception as exc:
            self.logger.warning("[ADAPTIVE_SESSION] Failed to format conversation history: %s", exc)

        return history

    def _track_conversation_turn(
        self,
        session: Optional[object],
        user_query: str,
        route: str,
        problems: list,
        emotions: list,
        response: str,
    ) -> None:
        """
        Track a new conversation turn in the session.
        
        Args:
            session: SessionMemory object
            user_query: User's query
            route: Selected route
            problems: Identified problems
            emotions: Detected emotions
            response: Generated response
        """
        if not session or not ConversationTurn:
            return

        try:
            turn = ConversationTurn(
                turn_num=session.total_turns + 1,
                user_query=user_query,
                route=route,
                problems=[{"problem": str(p)} for p in problems[:3]],
                emotions=[str(e) for e in emotions[:3]],
                response=response[:500],  # Truncate long responses
            )
            # Pass config value for max turns limit
            session.add_turn(turn, max_stored_turns=self.config.session_max_stored_turns)
            # Also update explicit last_resolved_query if present
            try:
                if hasattr(session, 'last_resolved_query'):
                    session.last_resolved_query = user_query
                if hasattr(session, 'last_route'):
                    session.last_route = route
            except Exception:
                pass
            self.logger.debug("[ADAPTIVE_SESSION] Added turn %d to session", turn.turn_num)
        except Exception as exc:
            self.logger.warning("[ADAPTIVE_SESSION] Failed to track turn: %s", exc)
