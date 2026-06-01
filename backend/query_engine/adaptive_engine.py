"""Route-aware orchestration for the adaptive GitaWise backend."""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Optional

logger = logging.getLogger(__name__)
logger.info("[ADAPTIVE] Module starting...")

from .config import QueryEngineConfig, get_logger, load_query_engine_config
logger.info("[ADAPTIVE] Config imported")
from .engine import GitaQueryEngine
logger.info("[ADAPTIVE] Engine imported")
from .generator import DirectResponseGenerator
logger.info("[ADAPTIVE] Generator imported")
from .lightweight_router import LightweightRouter
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

    POSITIVE_FEEDBACK_PHRASES = {
        "good",
        "great",
        "nice",
        "ok",
        "okay",
        "thanks",
        "thank you",
        "perfect",
        "got it",
        "understood",
        "goof",
        "goood",
        "gd",
        "thumbs up",
        "thumbsup",
        "thankyou",
    }
    NEGATIVE_FEEDBACK_PHRASES = {
        "bad",
        "wrong",
        "no",
        "not helpful",
        "didn't understand",
        "didnt understand",
        "that's not what i meant",
        "thats not what i meant",
        "irrelevant",
    }
    CONTINUATION_PHRASES = {
        "and",
        "and?",
        "then",
        "then?",
        "continue",
        "go on",
        "more",
        "what else",
        "elaborate",
        "tell me more",
        "next",
        "keep going",
    }
    CLARIFICATION_PHRASES = {
        "what do you mean",
        "explain",
        "how",
        "why",
    }
    CORRECTION_PHRASES = {
        "i asked about",
        "that's not what i asked",
        "thats not what i asked",
        "no i meant",
        "no, i meant",
        "you answered the wrong question",
        "my question was about",
    }
    DOMAIN_NOUNS = {
        "chapter",
        "verse",
        "shloka",
        "way",
        "ways",
        "gita",
        "karma",
        "dharma",
        "bhakti",
        "arjuna",
        "krishna",
        "teaching",
        "teachings",
    }
    REPEAT_REQUEST_PHRASES = {
        "repeat that",
        "say that again",
        "can you restate",
    }
    NON_ANSWER_PATTERNS = {
        "would you like to go deeper",
        "would you like to explore something new",
        "could you rephrase",
        "point to the part",
        "would you like to go deeper into that",
    }

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
            self.transition_manager = StateTransitionManager(self.query_engine.groq_client)
        except Exception:
            self.transition_manager = None

    def answer(self, user_query: str) -> AdaptiveAnswer:
        if not user_query.strip():
            raise ValueError("user_query must not be empty.")

        intent = self._detect_intent(user_query, [])
        if intent in {"FEEDBACK_POSITIVE", "FEEDBACK_NEGATIVE"}:
            return AdaptiveAnswer(
                original_query=user_query,
                route="philosophical_guidance",
                answer=self.direct_generator.generate_feedback_response(intent),
                cited_verses=[],
                contexts=[],
                warnings=[],
                used_rag=False,
            )

        route_result = self.query_engine.route_query(user_query)
        route = route_result.route
        self.logger.info("Adaptive route selected: %s", route)

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
            intent_history = conversation_history + [
                {"role": "system", "content": f"last_topic:{getattr(session, 'last_topic', '') or ''}"}
            ]
            self.logger.info("[ADAPTIVE_SESSION] Using session %s with %d prior turns", session_id, len(session.recent_turns))
        except Exception as exc:
            self.logger.warning("[ADAPTIVE_SESSION] Failed to load session: %s, falling back to stateless", exc)
            return self.answer(user_query)

        intent = self._detect_intent(user_query, intent_history)
        self.logger.info("[ADAPTIVE_SESSION] Intent detected: %s", intent)

        if intent in {"FEEDBACK_POSITIVE", "FEEDBACK_NEGATIVE"}:
            answer = AdaptiveAnswer(
                original_query=user_query,
                route="philosophical_guidance",
                answer=self.direct_generator.generate_feedback_response(intent),
                cited_verses=[],
                contexts=[],
                warnings=[],
                used_rag=False,
            )
            try:
                self._track_conversation_turn(
                    session=session,
                    user_query=user_query,
                    route="philosophical_guidance",
                    problems=[],
                    emotions=[],
                    response=answer.answer,
                    topic=getattr(session, "last_topic", None),
                    intent=intent,
                )
                self.session_cache.save_session(session)
            except Exception as exc:
                self.logger.warning("[ADAPTIVE_SESSION] Failed to track feedback turn: %s", exc)
            return answer

        # Build structured dialogue state from the session and classify transition
        dialogue_state = DialogueState.from_session(session)
        resolved_query = user_query
        is_continuation = False

        try:
            classification = None
            if intent == "CORRECTION":
                corrected_topic = self._extract_correction_topic(user_query) or user_query
                resolved_query = corrected_topic
                dialogue_state.active_topic = corrected_topic
                session.last_topic = corrected_topic
                conversation_history = []
                classification = {"transition_type": "topic_shift", "confidence": 0.99, "reason": "correction_preprocessor", "topic": corrected_topic}
            elif self._has_topic_router_context(dialogue_state) and self.transition_manager:
                classification = self.transition_manager.classify_transition(user_query, dialogue_state, conversation_history)
            elif intent == "TOPIC_SHIFT":
                dialogue_state.active_topic = user_query
                session.last_topic = user_query
                conversation_history = []
                classification = {"transition_type": "topic_shift", "confidence": 0.98, "reason": "intent_preprocessor", "topic": user_query}
            elif intent == "CONTINUATION":
                classification = {"transition_type": "continuation", "confidence": 0.99, "reason": "intent_preprocessor"}
            elif intent == "CLARIFICATION":
                classification = {"transition_type": "continuation", "confidence": 0.95, "reason": "clarification_preprocessor"}
            elif intent == "RESTATEMENT":
                restated_query = self._find_restatement_target(user_query, conversation_history) or user_query
                resolved_query = restated_query
                classification = {"transition_type": "topic_shift", "confidence": 0.97, "reason": "restatement_preprocessor", "topic": restated_query}
            elif self.transition_manager:
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
                    resolved_query = self._query_from_detected_topic(user_query, topic)
                dialogue_state.last_retrieved_verses = []
                if hasattr(session, "last_retrieved_verses"):
                    session.last_retrieved_verses = []
                conversation_history = []
                # rewrite conservatively: if rewriter exists, allow it to craft a retrieval-friendly query
                reason = str(classification.get("reason", ""))
                if intent in {"RESTATEMENT", "TOPIC_SHIFT", "CORRECTION"} or reason.startswith("llm_topic_shift_detector:"):
                    resolved_query = resolved_query
                elif self.contextual_rewriter:
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

        # Generate answer with conversation context.
        # Same-topic continuations reuse the topic's original retrieved verses.
        # Topic shifts are the only path that should retrieve fresh verses.
        ttype = classification.get("transition_type") if 'classification' in locals() else "topic_shift"
        should_reuse_cache = is_continuation and ttype == "continuation"
        
        if should_reuse_cache and dialogue_state.last_retrieved_verses:
            # CONTINUATION: Reuse cached retrieval from prior turn on same topic
            self.logger.info("[ADAPTIVE_SESSION] CONTINUATION: Reusing cached retrieval for topic=%s", dialogue_state.active_topic)
            problems = [Problem(problem=dialogue_state.active_topic or resolved_query.strip())]
            emotions = [EmotionResult(problem=dialogue_state.active_topic or resolved_query.strip(), emotion="none")]
            cached_contexts = self._deserialize_cached_contexts(dialogue_state.last_retrieved_verses)

            if cached_contexts:
                answer = self._run_grounded_route_with_cached_context(
                    user_query=resolved_query,
                    route=route,
                    problems=problems,
                    emotions=emotions,
                    cached_contexts=cached_contexts,
                    conversation_history=conversation_history,
                )
            else:
                self.logger.warning("[ADAPTIVE_SESSION] Cached topic contexts were empty/unusable; running fresh retrieval")
                problems, emotions = self.query_engine.analyze_query(resolved_query)
                retrieval_queries = self.query_engine.build_queries(emotions)
                answer = self._run_grounded_route(
                    user_query=resolved_query,
                    route=route,
                    problems=problems,
                    emotions=emotions,
                    retrieval_queries=retrieval_queries,
                    conversation_history=conversation_history,
                    session_id=session_id,
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
                session_id=session_id,
            )
        else:
            # TOPIC_SHIFT or fresh query: Do full analysis + retrieval
            if ttype == "topic_shift":
                self.logger.info("[ADAPTIVE_SESSION] TOPIC_SHIFT: Running fresh analysis and retrieval for new topic")
            problems, emotions = self.query_engine.analyze_query(resolved_query)
            retrieval_queries = self.query_engine.build_queries(emotions)
            answer = self._run_grounded_route(
                user_query=resolved_query,
                route=route,
                problems=problems,
                emotions=emotions,
                retrieval_queries=retrieval_queries,
                conversation_history=conversation_history,
                session_id=session_id,
            )

        if intent == "CORRECTION":
            answer.answer = f"Apologies, let me address that.\n\n{answer.answer}"

        # Track this turn in session and update structured dialogue state
        try:
            self._track_conversation_turn(
                session=session,
                user_query=user_query,
                route=route,
                problems=problems,
                emotions=[],
                response=answer.answer,
                topic=getattr(dialogue_state, "active_topic", None),
                intent=intent,
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

            if not topic:
                topic = self._topic_from_query(resolved_query)
            if topic:
                state.active_topic = topic
            
            # Store retrieved contexts in dialogue state for reuse on continuations (same topic)
            try:
                if answer.used_rag and getattr(answer, 'contexts', None):
                    state.last_retrieved_verses = [
                        c.model_dump() if hasattr(c, 'model_dump') else c
                        for c in answer.contexts
                    ]
                    self.logger.info("[ADAPTIVE_SESSION] Stored %d contexts in dialogue state for continuation", len(state.last_retrieved_verses))
            except Exception as exc:
                self.logger.warning("[ADAPTIVE_SESSION] Failed to store contexts in dialogue state: %s", exc)
            
            state.active_emotions = getattr(session, 'detected_emotions', []) or []
            state.apply_to_session(session)
            self.session_cache.save_session(session)
            self.logger.debug("[ADAPTIVE_SESSION] Updated dialogue state for session %s", session_id)
        except Exception as exc:
            self.logger.warning("[ADAPTIVE_SESSION] Failed to update dialogue state: %s", exc)

        return answer

    def _detect_intent(self, message: str, history: list[dict[str, str]] | None = None) -> str:
        """Classify the incoming turn before route selection or retrieval begins."""
        raw = (message or "").strip()
        if not raw:
            return "GENUINE_QUERY"

        lowered = raw.lower()
        normalized = self._normalize_intent_text(raw)
        tokens = self._intent_tokens(raw)
        word_count = len(tokens)
        has_history = bool(history)
        has_question_mark = "?" in raw

        if normalized in {self._normalize_intent_text(item) for item in self.REPEAT_REQUEST_PHRASES}:
            return "GENUINE_QUERY"

        if self._matches_phrase(normalized, self.CORRECTION_PHRASES):
            return "CORRECTION"

        if has_history and self._is_restatement(raw, history):
            return "RESTATEMENT"

        if self._is_positive_feedback(raw) and not self._contains_noun_or_number_signal(raw, tokens):
            return "FEEDBACK_POSITIVE"
        if self._matches_phrase(normalized, self.NEGATIVE_FEEDBACK_PHRASES) and not self._contains_noun_or_number_signal(raw, tokens):
            return "FEEDBACK_NEGATIVE"
        if self._matches_phrase(normalized, self.CONTINUATION_PHRASES):
            return "CONTINUATION"
        if has_history and self._references_previous_answer(normalized):
            return "CONTINUATION"
        if self._is_clarification_request(raw, has_history):
            return "CLARIFICATION"
        if has_history and self._is_topic_shift(raw, history):
            return "TOPIC_SHIFT"

        if word_count == 1:
            token = tokens[0]
            if self._contains_noun_or_number_signal(raw, tokens):
                return "GENUINE_QUERY"
            if LightweightRouter.is_gita_concept_word(token):
                return "GENUINE_QUERY"
            if LightweightRouter.fuzzy_match_short_token(token, LightweightRouter.SHORT_POSITIVE_WORDS):
                return "FEEDBACK_POSITIVE"
            if LightweightRouter.fuzzy_match_short_token(token, LightweightRouter.SHORT_NEGATIVE_WORDS):
                return "FEEDBACK_NEGATIVE"
            if LightweightRouter.fuzzy_match_short_token(token, LightweightRouter.SHORT_CONTINUATION_WORDS):
                return "CONTINUATION"

        if has_history and word_count <= 3 and not has_question_mark:
            if self._contains_noun_or_number_signal(raw, tokens):
                return "GENUINE_QUERY"
            if any(token in {"more", "next", "continue", "elaborate", "then", "and"} for token in tokens):
                return "CONTINUATION"
            return "FEEDBACK_POSITIVE"

        return "GENUINE_QUERY"

    @staticmethod
    def _normalize_intent_text(text: str) -> str:
        normalized = text.lower()
        normalized = re.sub(r"\s+", " ", normalized)
        normalized = re.sub(r"[^\w\s?']", " ", normalized)
        return re.sub(r"\s+", " ", normalized).strip()

    @staticmethod
    def _intent_tokens(text: str) -> list[str]:
        return re.findall(r"\b[\w']+\b", text.lower())

    def _is_positive_feedback(self, text: str) -> bool:
        if "👍" in text:
            return True
        normalized = self._normalize_intent_text(text)
        if self._matches_phrase(normalized, self.POSITIVE_FEEDBACK_PHRASES):
            return True
        compact = LightweightRouter.normalize_short_text(text)
        if compact and LightweightRouter.fuzzy_match_short_token(compact, LightweightRouter.SHORT_POSITIVE_WORDS):
            return True
        return False

    @staticmethod
    def _matches_phrase(normalized_text: str, phrases: set[str]) -> bool:
        normalized_phrases = {re.sub(r"\s+", " ", phrase.lower()).strip() for phrase in phrases}
        if normalized_text in normalized_phrases:
            return True
        return LightweightRouter.fuzzy_match_text(normalized_text, normalized_phrases) is not None

    def _is_clarification_request(self, text: str, has_history: bool) -> bool:
        if not has_history:
            return False
        normalized = self._normalize_intent_text(text)
        if any(
            normalized == phrase or normalized.startswith(f"{phrase} ")
            for phrase in self.CLARIFICATION_PHRASES
        ):
            return True
        if "?" in text or len(self._intent_tokens(text)) <= 6:
            lowered = text.lower()
            if any(marker in lowered for marker in {"that", "this", "it", "you mean", "why", "how"}):
                return True
        return False

    def _is_topic_shift(self, text: str, history: list[dict[str, str]]) -> bool:
        last_topic = self._extract_last_topic(history)
        if not last_topic:
            return False
        current_nouns = self._subject_tokens(text)
        last_topic_nouns = self._subject_tokens(last_topic)
        if not current_nouns or not last_topic_nouns:
            return False
        overlap = current_nouns.intersection(last_topic_nouns)
        if "?" in text and not overlap and self._introduces_new_subject_noun(text, history):
            return True
        return not overlap

    def _extract_correction_topic(self, text: str) -> str | None:
        patterns = [
            r"i asked about\s+(.+)$",
            r"my question was about\s+(.+)$",
            r"no,?\s+i meant\s+(.+)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip(" .?!")
        return None

    @staticmethod
    def _has_topic_router_context(dialogue_state: DialogueState) -> bool:
        return bool(
            getattr(dialogue_state, "active_topic", None)
            and getattr(dialogue_state, "last_retrieved_verses", None)
        )

    def _topic_from_query(self, text: str) -> str:
        """Extract a compact topic label from the current user/resolved query."""
        normalized = self._normalize_intent_text(text)
        patterns = [
            r"what is (.+)$",
            r"explain (.+)$",
            r"tell me about (.+)$",
            r"what does the gita say about (.+)$",
            r"about (.+)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, normalized, re.IGNORECASE)
            if match:
                return self._clean_topic_label(match.group(1))
        return self._clean_topic_label(normalized)

    def _query_from_detected_topic(self, original_query: str, detected_topic: str) -> str:
        """Build a standalone retrieval query for a newly detected topic."""
        topic = self._clean_topic_label(detected_topic)
        if not topic:
            return original_query

        normalized_original = self._normalize_intent_text(original_query)
        if topic in normalized_original:
            return original_query
        return f"What is {topic} according to the Bhagavad Gita?"

    @staticmethod
    def _clean_topic_label(text: str) -> str:
        cleaned = re.sub(r"\b(?:bhagavad|gita|krishna|arjuna|teachings?|verses?|concept|meaning)\b", " ", text, flags=re.IGNORECASE)
        cleaned = re.sub(r"[^a-z0-9\s'-]+", " ", cleaned.lower())
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" .?!")
        return cleaned[:80] or text.strip()[:80]

    @staticmethod
    def _extract_last_topic(history: list[dict[str, str]]) -> str:
        for item in reversed(history):
            if item.get("role") == "system" and str(item.get("content", "")).startswith("last_topic:"):
                return str(item.get("content", "")).split("last_topic:", 1)[1].strip()
        return ""

    def _subject_tokens(self, text: str) -> set[str]:
        ignored = {
            "what", "why", "how", "when", "where", "who", "about", "mean", "asked",
            "question", "that", "this", "your", "with", "from", "into", "then",
            "more", "continue", "please", "tell",
        }
        return {
            token for token in self._intent_tokens(text)
            if token not in ignored and len(token) > 2 and not token.isdigit()
        }

    def _introduces_new_subject_noun(self, text: str, history: list[dict[str, str]]) -> bool:
        current_tokens = self._subject_tokens(text)
        recent_text = " ".join(
            item.get("content", "") for item in history[-4:] if item.get("role") in {"user", "assistant"}
        )
        prior_tokens = self._subject_tokens(recent_text)
        return bool(current_tokens - prior_tokens)

    def _contains_noun_or_number_signal(self, text: str, tokens: list[str]) -> bool:
        if any(char.isdigit() for char in text):
            return True
        if any(token in self.DOMAIN_NOUNS for token in tokens):
            return True
        if len(tokens) <= 3:
            filler = {
                "good", "great", "nice", "ok", "okay", "thanks", "thank", "you",
                "perfect", "got", "it", "understood", "bad", "wrong", "not",
                "helpful", "and", "then", "more", "continue", "what", "else",
            }
            content_tokens = [token for token in tokens if token not in filler]
            if any(len(token) > 4 for token in content_tokens):
                return True
        return False

    @staticmethod
    def _references_previous_answer(normalized_text: str) -> bool:
        previous_markers = {
            "what you just said",
            "what you said",
            "previous answer",
            "last answer",
            "your answer",
            "that answer",
            "above answer",
        }
        action_markers = {
            "summarize",
            "summary",
            "explain",
            "example",
            "elaborate",
            "rephrase",
            "short",
            "one sentence",
        }
        return any(marker in normalized_text for marker in previous_markers) and any(
            marker in normalized_text for marker in action_markers
        )

    def _is_restatement(self, text: str, history: list[dict[str, str]]) -> bool:
        prior_question = self._find_restatement_target(text, history)
        if not prior_question:
            return False
        last_answer = self._last_assistant_message(history)
        if not last_answer:
            return False
        return not self._did_last_answer_address(prior_question, last_answer)

    def _find_restatement_target(self, text: str, history: list[dict[str, str]]) -> str | None:
        normalized_current = self._normalize_similarity_text(text)
        if not normalized_current:
            return None

        best_match: str | None = None
        best_score = 0.0
        user_messages = [item.get("content", "") for item in history if item.get("role") == "user"]
        for candidate in reversed(user_messages[-6:]):
            normalized_candidate = self._normalize_similarity_text(candidate)
            if not normalized_candidate:
                continue
            score = SequenceMatcher(None, normalized_current, normalized_candidate).ratio()
            overlap = self._token_overlap_ratio(normalized_current, normalized_candidate)
            if score >= 0.84 or overlap >= 0.72:
                combined = score + overlap
                if combined > best_score:
                    best_score = combined
                    best_match = candidate
        return best_match

    def _did_last_answer_address(self, question: str, answer: str) -> bool:
        normalized_answer = self._normalize_intent_text(answer)
        if any(pattern in normalized_answer for pattern in self.NON_ANSWER_PATTERNS):
            return False
        question_tokens = self._meaningful_tokens(question)
        answer_tokens = self._meaningful_tokens(answer)
        if not question_tokens or not answer_tokens:
            return False
        overlap = len(question_tokens.intersection(answer_tokens)) / max(1, len(question_tokens))
        if overlap >= 0.35:
            return True
        if len(answer_tokens) >= 20 and overlap >= 0.22:
            return True
        return False

    @staticmethod
    def _last_assistant_message(history: list[dict[str, str]]) -> str | None:
        for item in reversed(history):
            if item.get("role") == "assistant" and item.get("content"):
                return item["content"]
        return None

    @staticmethod
    def _normalize_similarity_text(text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]+", " ", text.lower())).strip()

    def _meaningful_tokens(self, text: str) -> set[str]:
        stopwords = {
            "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "is",
            "are", "what", "why", "how", "do", "does", "tell", "me", "about",
            "please", "can", "you", "i", "it", "this", "that",
        }
        return {token for token in self._intent_tokens(text) if token not in stopwords and len(token) > 2}

    def _token_overlap_ratio(self, text_a: str, text_b: str) -> float:
        tokens_a = self._meaningful_tokens(text_a)
        tokens_b = self._meaningful_tokens(text_b)
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a.intersection(tokens_b)) / max(1, len(tokens_a.union(tokens_b)))

    def _run_grounded_route(
        self,
        user_query: str,
        route: str,
        problems: list[Problem],
        emotions: list[EmotionResult],
        retrieval_queries: list[RetrievalQuery],
        conversation_history: list[dict[str, str]] | None = None,
        session_id: str | None = None,
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
            retrieved_verses = self._retrieve_with_session_cache(
                retrieval_queries=retrieval_queries,
                session_id=session_id,
            )
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
            fallback_contexts = self._select_topic_seed_contexts(retrieved_verses)
            if not fallback_contexts:
                return self.direct_generator.generate(
                    user_query=user_query,
                    route=route,
                    warnings=warnings,
                    fallback_note="Retrieved verse relevance was too weak for trustworthy grounding. Respond without fake citations.",
                    conversation_history=conversation_history,
                )

            warnings.append("used_low_confidence_topic_seed")
            generated = self.query_engine.generator.generate(
                user_query=user_query,
                problems=problems,
                emotions=emotions,
                contexts=fallback_contexts,
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

    def _retrieve_with_session_cache(
        self,
        retrieval_queries: list[RetrievalQuery],
        session_id: str | None,
    ) -> list[RetrievedVerse]:
        """Retrieve verses, using the Supabase-backed cache when a session is available."""
        if not retrieval_queries:
            return []

        if self.session_cache:
            cached_verses = None
            for ret_q in retrieval_queries:
                cached = self.session_cache.get_cached_retrieval(ret_q.query)
                if cached:
                    cached_verses = cached
                    self.logger.info("[ADAPTIVE_SESSION] Cache hit: %d retrieved verses for '%s'", len(cached), ret_q.query)
                    break

            if cached_verses:
                return [
                    RetrievedVerse(**verse) if isinstance(verse, dict) else verse
                    for verse in cached_verses
                ]

        retrieved_verses = self.query_engine.retrieve(retrieval_queries)

        if self.session_cache and session_id and retrieved_verses:
            serialized = [
                verse.model_dump() if hasattr(verse, "model_dump") else verse
                for verse in retrieved_verses
            ]
            for ret_q in retrieval_queries:
                self.session_cache.cache_retrieval(session_id, ret_q.query, serialized)
            self.logger.info(
                "[ADAPTIVE_SESSION] Cached %d retrieved verses for %d retrieval queries",
                len(serialized),
                len(retrieval_queries),
            )

        return retrieved_verses

    def _select_topic_seed_contexts(self, retrieved_verses: list[RetrievedVerse]) -> list[RetrievedVerse]:
        """Pick the best available retrieved verses as the reusable topic seed."""
        if not retrieved_verses:
            return []

        ranked = sorted(
            retrieved_verses,
            key=lambda verse: (verse.retrieval_score or verse.score or 0.0),
            reverse=True,
        )[: self.config.generation_context_top_k]
        return self.query_engine.build_context(ranked)

    def _deserialize_cached_contexts(self, cached_verses: list[object]) -> list[RetrievedVerse]:
        """Convert cached verse payloads back into RetrievedVerse objects."""
        contexts: list[RetrievedVerse] = []
        for verse in cached_verses:
            try:
                contexts.append(RetrievedVerse(**verse) if isinstance(verse, dict) else verse)
            except Exception as exc:
                self.logger.warning("[ADAPTIVE_SESSION] Skipping invalid cached verse: %s", exc)
        return contexts

    def _run_grounded_route_with_cached_context(
        self,
        user_query: str,
        route: str,
        problems: list[Problem],
        emotions: list[EmotionResult],
        cached_contexts: list[RetrievedVerse],
        conversation_history: list[dict[str, str]] | None = None,
    ) -> AdaptiveAnswer:
        """
        Run grounded route using cached retrieved contexts (for continuations on same topic).
        
        Args:
            user_query: Current user query
            route: Selected route
            problems: Identified problems
            emotions: Detected emotions
            cached_contexts: Pre-retrieved and cached contexts from prior turn
            conversation_history: Optional previous conversation turns
        """
        warnings: list[str] = ["used_cache:retrieval"]
        self.logger.info("[ADAPTIVE] Using cached contexts for continuation: %d verses", len(cached_contexts))
        
        try:
            generated = self.query_engine.generator.generate(
                user_query=user_query,
                problems=problems,
                emotions=emotions,
                contexts=cached_contexts,
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
        except Exception as exc:
            self.logger.exception("Failed to generate using cached contexts: %s", exc)
            warnings.append(f"cached_generation_failed: {exc}")
            return self.direct_generator.generate(
                user_query=user_query,
                route=route,
                warnings=warnings,
                fallback_note="Failed to use cached context. Respond directly and cautiously.",
                conversation_history=conversation_history,
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
        topic: str | None = None,
        intent: str | None = None,
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
                topic=topic,
                last_answer=response[:500],
                user_intent=intent,
                turn_count=session.total_turns + 1,
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
                if hasattr(session, 'last_answer'):
                    session.last_answer = response[:500]
                if hasattr(session, 'last_user_intent'):
                    session.last_user_intent = intent
                if hasattr(session, 'turn_count'):
                    session.turn_count = session.total_turns
                if hasattr(session, 'last_topic') and topic:
                    session.last_topic = topic
            except Exception:
                pass
            self.logger.debug("[ADAPTIVE_SESSION] Added turn %d to session", turn.turn_num)
        except Exception as exc:
            self.logger.warning("[ADAPTIVE_SESSION] Failed to track turn: %s", exc)
