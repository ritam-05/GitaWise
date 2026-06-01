from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from .continuation_detector import ContinuationDetector
from .dialogue_state import DialogueState
from .prompts import render_topic_shift_detection_prompt

LOGGER = logging.getLogger(__name__)


class StateTransitionManager:
    """Detects continuation, topic shifts, and emotional escalations.

    Lightweight, rule-based implementation with simple semantic checks.
    """

    EMOTION_ESCALATION_KEYPHRASES = [
        "i hate my life",
        "nothing matters",
        "i feel broken",
        "i feel empty",
        "i cannot continue",
        "i can't continue",
        "i want to die",
        "kill myself",
        "worthless",
        "suicide",
        "i am done",
    ]

    TOPIC_SHIFT_PATTERNS = [
        r"tell me about (.+)",
        r"what does the gita say about (.+)",
        r"what about (.+)",
        r"tell me (.+)",
    ]

    def __init__(self, groq_client: object | None = None, min_words_for_continuation: int = 4) -> None:
        self.groq_client = groq_client
        self.detector = ContinuationDetector(min_words=min_words_for_continuation)

    def classify_transition(
        self,
        user_query: str,
        dialogue_state: DialogueState | None = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Return a classification dict with keys: transition_type, confidence, details."""
        q = (user_query or "").strip().lower()

        # 1. Check for emotional escalation phrases (highest priority)
        for phrase in self.EMOTION_ESCALATION_KEYPHRASES:
            if phrase in q:
                return {"transition_type": "emotion_shift", "confidence": 0.95, "reason": f"matched_phrase:{phrase}"}

        llm_decision = self.detect_topic_shift_with_llm(user_query, dialogue_state)
        if llm_decision:
            return llm_decision

        # 2. Check for explicit topic shift patterns
        for pat in self.TOPIC_SHIFT_PATTERNS:
            m = re.search(pat, q)
            if m:
                topic = m.group(1).strip()
                if self._has_topic_overlap(topic, dialogue_state):
                    return {
                        "transition_type": "continuation",
                        "confidence": 0.82,
                        "reason": f"pattern_overlap:{pat}",
                        "topic": getattr(dialogue_state, "active_topic", None),
                    }
                return {"transition_type": "topic_shift", "confidence": 0.9, "reason": f"pattern:{pat}", "topic": topic}

        # 3. Continuation detection
        if self.detector.is_continuation(user_query, conversation_history):
            return {"transition_type": "continuation", "confidence": 0.8, "reason": "heuristic_continuation"}

        # 4. Heuristic semantic shift: check token overlap with prior topic
        if dialogue_state and dialogue_state.active_topic and q:
            # simple token overlap ratio (Jaccard-like)
            prior_tokens = set(re.findall(r"\b[\w']+\b", (dialogue_state.active_topic or "").lower()))
            curr_tokens = set(re.findall(r"\b[\w']+\b", q))
            if prior_tokens and curr_tokens:
                inter = prior_tokens.intersection(curr_tokens)
                union = prior_tokens.union(curr_tokens)
                score = len(inter) / max(1, len(union))
                if score >= 0.25:
                    return {"transition_type": "continuation", "confidence": 0.7, "reason": "token_overlap", "overlap": score}
                if score < 0.2:
                    # low overlap suggests possible topic shift
                    return {"transition_type": "topic_shift", "confidence": 0.6, "reason": "low_overlap", "overlap": score}

        # Default: treat as topic shift with low confidence if query is substantive
        tokens = re.findall(r"\b[\w']+\b", q)
        if len(tokens) >= 3:
            return {"transition_type": "topic_shift", "confidence": 0.5, "reason": "default_substantive"}

        # Fallback: continuation
        return {"transition_type": "continuation", "confidence": 0.4, "reason": "fallback"}

    def detect_topic_shift_with_llm(
        self,
        user_query: str,
        dialogue_state: DialogueState | None,
    ) -> Dict[str, Any] | None:
        """Use the LLM as the only primary judge for continuation vs topic shift."""
        if not self.groq_client or not dialogue_state:
            return None

        current_topic = (dialogue_state.active_topic or "").strip()
        cached_context = dialogue_state.last_retrieved_verses or []
        if not current_topic or not cached_context:
            return None

        try:
            prompt = render_topic_shift_detection_prompt(
                current_topic=current_topic,
                context_summary=self._context_summary(cached_context),
                user_message=user_query,
            )
            payload = self.groq_client.invoke_json(prompt)
            decision = str(payload.get("decision", "")).strip().upper()
            detected_topic = str(payload.get("detected_topic", "")).strip()
            reason = str(payload.get("reason", "")).strip() or "topic_shift_detector"

            if decision == "CONTINUATION":
                return {
                    "transition_type": "continuation",
                    "confidence": 0.9,
                    "reason": f"llm_topic_shift_detector:{reason}",
                    "topic": current_topic,
                    "detected_topic": detected_topic or current_topic,
                }
            if decision == "TOPIC_SHIFT":
                return {
                    "transition_type": "topic_shift",
                    "confidence": 0.9,
                    "reason": f"llm_topic_shift_detector:{reason}",
                    "topic": detected_topic or user_query,
                    "detected_topic": detected_topic or user_query,
                }
        except Exception as exc:
            LOGGER.warning("LLM topic-shift detection failed; falling back to heuristics: %s", exc)

        return None

    @staticmethod
    def _context_summary(cached_context: list[object]) -> str:
        summaries: list[str] = []
        for item in cached_context[:5]:
            if isinstance(item, dict):
                chapter = item.get("chapter")
                verse = item.get("verse")
                label = f"{chapter}.{verse}" if chapter and verse else "cached verse"
                text = (
                    item.get("summary")
                    or item.get("interpretation")
                    or item.get("translation")
                    or item.get("retrieval_text")
                    or ""
                )
                topics = item.get("topics") or []
            else:
                chapter = getattr(item, "chapter", None)
                verse = getattr(item, "verse", None)
                label = f"{chapter}.{verse}" if chapter and verse else "cached verse"
                text = (
                    getattr(item, "summary", "")
                    or getattr(item, "interpretation", "")
                    or getattr(item, "translation", "")
                    or getattr(item, "retrieval_text", "")
                )
                topics = getattr(item, "topics", []) or []

            topic_text = ", ".join(str(topic) for topic in topics[:4]) if isinstance(topics, list) else str(topics)
            summary = f"{label}: {str(text).strip()[:220]}"
            if topic_text:
                summary = f"{summary} Topics: {topic_text}"
            summaries.append(summary)

        return "\n".join(summaries) if summaries else "No cached context available."

    @staticmethod
    def _has_topic_overlap(topic: str, dialogue_state: DialogueState | None) -> bool:
        if not dialogue_state or not dialogue_state.active_topic:
            return False

        ignored = {
            "is",
            "are",
            "was",
            "were",
            "what",
            "who",
            "about",
            "tell",
            "me",
            "the",
            "a",
            "an",
            "gita",
            "say",
            "says",
            "does",
            "do",
            "explain",
            "define",
        }
        topic_tokens = {
            token
            for token in re.findall(r"\b[\w']+\b", topic.lower())
            if token not in ignored and len(token) > 2
        }
        active_tokens = {
            token
            for token in re.findall(r"\b[\w']+\b", dialogue_state.active_topic.lower())
            if token not in ignored and len(token) > 2
        }
        return bool(topic_tokens.intersection(active_tokens))
