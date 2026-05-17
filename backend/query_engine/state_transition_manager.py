from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .continuation_detector import ContinuationDetector
from .dialogue_state import DialogueState


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

    def __init__(self, min_words_for_continuation: int = 4) -> None:
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

        # 2. Check for explicit topic shift patterns
        for pat in self.TOPIC_SHIFT_PATTERNS:
            m = re.search(pat, q)
            if m:
                topic = m.group(1).strip()
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
                if score >= 0.5:
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
