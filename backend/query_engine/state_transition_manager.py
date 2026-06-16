from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from .dialogue_state import DialogueState
from .prompts import render_topic_shift_detection_prompt

LOGGER = logging.getLogger(__name__)


class StateTransitionManager:
    """Detects continuation vs topic shifts using the LLM as the sole judge.

    The only hard rule is emotional-escalation detection (safety-critical).
    Everything else is decided by the LLM topic-shift detector.
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

    def __init__(self, groq_client: object | None = None, min_words_for_continuation: int = 4) -> None:
        self.groq_client = groq_client

    def classify_transition(
        self,
        user_query: str,
        dialogue_state: DialogueState | None = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Return a classification dict with keys: transition_type, confidence, reason.

        Decision hierarchy:
        1. Emotional escalation (hard rule — safety-critical).
        2. LLM topic-shift detector (sole judge for continuation vs topic shift).
        3. If LLM unavailable or no active topic → default to topic_shift (safe fallback).
        """
        q = (user_query or "").strip().lower()

        # 1. Emotional escalation — always overrides LLM (safety)
        for phrase in self.EMOTION_ESCALATION_KEYPHRASES:
            if phrase in q:
                return {
                    "transition_type": "emotion_shift",
                    "confidence": 0.97,
                    "reason": f"emotion_escalation_phrase:{phrase}",
                }

        # 2. LLM is the sole topic-shift judge
        llm_decision = self.detect_topic_shift_with_llm(
            user_query, dialogue_state, conversation_history
        )
        if llm_decision:
            return llm_decision

        # 3. Safe fallback: if no active topic or LLM unavailable, treat as topic shift
        # (fresh retrieval is always safer than answering without context)
        return {
            "transition_type": "topic_shift",
            "confidence": 0.5,
            "reason": "no_active_topic_or_llm_unavailable",
        }

    def detect_topic_shift_with_llm(
        self,
        user_query: str,
        dialogue_state: DialogueState | None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any] | None:
        """Use the LLM as the sole judge for continuation vs topic shift.

        Only requires an active_topic — cached verses are optional context.
        """
        if not self.groq_client or not dialogue_state:
            return None

        current_topic = (dialogue_state.active_topic or "").strip()
        if not current_topic:
            return None

        # Build context summary from cached verses OR conversation history
        cached_context = dialogue_state.last_retrieved_verses or []
        if cached_context:
            context_summary = self._context_summary(cached_context)
        elif conversation_history:
            context_summary = self._conversation_summary(conversation_history)
        else:
            context_summary = "No prior context available."

        try:
            prompt = render_topic_shift_detection_prompt(
                current_topic=current_topic,
                context_summary=context_summary,
                user_message=user_query,
            )
            payload = self.groq_client.invoke_json(prompt)
            decision = str(payload.get("decision", "")).strip().upper()
            detected_topic = str(payload.get("detected_topic", "")).strip()
            reason = str(payload.get("reason", "")).strip() or "llm_decision"

            if decision == "CONTINUATION":
                return {
                    "transition_type": "continuation",
                    "confidence": 0.9,
                    "reason": f"llm:{reason}",
                    "topic": current_topic,
                    "detected_topic": detected_topic or current_topic,
                }
            if decision == "TOPIC_SHIFT":
                return {
                    "transition_type": "topic_shift",
                    "confidence": 0.9,
                    "reason": f"llm:{reason}",
                    "topic": detected_topic or user_query,
                    "detected_topic": detected_topic or user_query,
                }

            LOGGER.warning(
                "[STM] LLM returned unexpected decision=%r; falling back to topic_shift", decision
            )
        except Exception as exc:
            LOGGER.warning("[STM] LLM topic-shift detection failed: %s", exc)

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

            topic_text = ", ".join(str(t) for t in topics[:4]) if isinstance(topics, list) else str(topics)
            summary = f"{label}: {str(text).strip()[:220]}"
            if topic_text:
                summary = f"{summary} | Topics: {topic_text}"
            summaries.append(summary)

        return "\n".join(summaries) if summaries else "No cached context available."

    @staticmethod
    def _conversation_summary(conversation_history: List[Dict[str, str]]) -> str:
        """Build a compact summary from recent conversation turns for the LLM prompt."""
        lines: list[str] = []
        for turn in conversation_history[-6:]:
            role = turn.get("role", "").upper()
            content = (turn.get("content") or "").strip()[:200]
            if content and role in {"USER", "ASSISTANT"}:
                lines.append(f"{role}: {content}")
        return "\n".join(lines) if lines else "No conversation history available."
