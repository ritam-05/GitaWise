"""Unified problem decomposition and emotion detection using LangChain chains."""

from __future__ import annotations

import json
import re

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from .config import QueryEngineConfig, get_logger
from .decomposer import GroqJSONClient
from .models import CombinedAnalysisResult, EmotionLabel, EmotionResult, Problem, ProblemWithEmotion
from .prompts import render_combined_analysis_prompt, ALLOWED_EMOTIONS

LOGGER = get_logger(__name__)


class CombinedAnalyzer:
    """
    Merges query decomposition and emotion detection into ONE efficient LLM call using LangChain.

    This replaces the separate decomposer + emotion_detector pipeline,
    reducing token usage and latency significantly.
    """

    def __init__(self, groq_client: GroqJSONClient | None = None, config: QueryEngineConfig | None = None) -> None:
        self.groq_client = groq_client
        self.config = config or QueryEngineConfig()
        self.logger = get_logger(self.__class__.__name__)
        self.llm = ChatGroq(
            api_key=self.config.groq_api_key,
            model_name=self.config.groq_analyzer_model_name,  # Use smaller model for extraction
            temperature=self.config.groq_temperature,
        )

    def analyze(self, query: str) -> tuple[list[Problem], list[EmotionResult]]:
        """
        Perform unified analysis: decompose query and detect emotions in one LangChain call.
        
        OPTIMIZED:
        - Uses smaller model (llama-3.1-8b-instant instead of 70b)
        - Compact, deterministic prompt (token-optimized)
        - Single LLM call instead of separate pipeline
        - Graceful fallback for failures

        Args:
            query: User's query string.

        Returns:
            Tuple of (problems list, emotion_results list) preserving original order.

        Raises:
            ValueError: If analysis fails after retries.
        """
        if not query.strip():
            raise ValueError("Query must not be empty.")

        # Single LLM call for both decomposition and emotion detection
        prompt_text = render_combined_analysis_prompt(query)
        
        messages = [
            SystemMessage(content="Return valid JSON only."),
            HumanMessage(content=prompt_text),
        ]
        
        response = self.llm.invoke(messages)
        raw_content = response.content or "{}"
        
        payload = self._parse_payload(raw_content, query)

        try:
            result = CombinedAnalysisResult.model_validate(payload)
        except ValidationError as exc:
            # Attempt to sanitize the payload by normalizing unknown emotions
            self.logger.warning("Invalid combined analysis payload, attempting sanitization: %s", payload)

            try:
                allowed_emotions = set(ALLOWED_EMOTIONS)

                sanitized = {**(payload or {})} if isinstance(payload, dict) else {}
                problems = sanitized.get("problems")
                if isinstance(problems, list):
                    new_problems = []
                    for p in problems:
                        if not isinstance(p, dict):
                            continue
                        probs = p.copy()
                        emotions = probs.get("emotions")
                        if isinstance(emotions, list):
                            cleaned = []
                            for e in emotions:
                                s = (e or "").strip().lower()
                                if s in allowed_emotions:
                                    if s not in cleaned:
                                        cleaned.append(s)
                                else:
                                    # map unknown to 'none' (avoid duplication)
                                    if "none" not in cleaned:
                                        cleaned.append("none")
                            probs["emotions"] = cleaned or ["none"]
                        else:
                            probs["emotions"] = ["none"]
                        # Ensure problem text exists
                        if not probs.get("problem"):
                            probs["problem"] = "generic informational query"
                        new_problems.append(probs)
                    sanitized["problems"] = new_problems

                # Retry validation with sanitized payload
                result = CombinedAnalysisResult.model_validate(sanitized)
                self.logger.info("Sanitized combined analysis payload validated successfully.")
            except Exception:
                self.logger.error("Sanitization failed for combined analysis payload: %s", payload)
                result = CombinedAnalysisResult.model_validate(self._fallback_payload(query))

        if not result.problems:
            self.logger.warning("Combined analyzer returned no problems for query: %s", query)
            # Fallback: treat as single generic query
            result.problems = [
                ProblemWithEmotion(
                    problem="generic informational query",
                    emotions=["none"],
                )
            ]

        # Convert combined results into separate Problem and EmotionResult lists
        problems = []
        emotion_results = []

        for item in result.problems:
            # Validate and normalize emotions
            normalized_emotions = self._normalize_emotions(item.emotions)

            # Create Problem record
            problems.append(Problem(problem=item.problem))

            # Create EmotionResult for each emotion (or single dominant one)
            # For compatibility with downstream, we pick the first emotion or "none"
            dominant_emotion = normalized_emotions[0] if normalized_emotions else "none"
            emotion_results.append(
                EmotionResult(
                    problem=item.problem,
                    emotion=dominant_emotion,
                )
            )

        self.logger.info(
            "Combined analysis: %d problems identified with emotions",
            len(problems),
        )
        for problem, emotion_result in zip(problems, emotion_results):
            self.logger.debug(
                "Problem: '%s' -> Emotion: '%s'",
                problem.problem,
                emotion_result.emotion,
            )

        return problems, emotion_results

    def _normalize_emotions(self, emotions: list[str]) -> list[EmotionLabel]:
        """
        Validate and normalize emotions from LLM response.

        Args:
            emotions: List of emotion strings from the LLM.

        Returns:
            List of valid EmotionLabel values, up to 3.
        """
        allowed_emotions = set(ALLOWED_EMOTIONS)

        normalized = []
        for emotion in emotions:
            cleaned = (emotion or "").strip().lower()
            if cleaned in allowed_emotions:
                if cleaned not in normalized:  # Avoid duplicates
                    normalized.append(cleaned)  # type: ignore
            else:
                self.logger.warning(
                    "Unknown emotion '%s' from LLM, using 'none' as fallback",
                    emotion,
                )
                if "none" not in normalized:
                    normalized.append("none")  # type: ignore

        # Cap at 3 emotions
        return normalized[:3]  # type: ignore

    def _parse_payload(self, raw_content: str, query: str) -> dict:
        """Parse LLM JSON, falling back instead of failing the request."""
        try:
            payload = json.loads(raw_content)
            return payload if isinstance(payload, dict) else self._fallback_payload(query)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", raw_content, re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group(0))
                return payload if isinstance(payload, dict) else self._fallback_payload(query)
            except json.JSONDecodeError as exc:
                self.logger.warning(
                    "Malformed combined analysis JSON, using fallback. error=%s raw=%s",
                    exc,
                    raw_content[:500],
                )
        else:
            self.logger.warning(
                "Combined analysis response did not contain JSON, using fallback. raw=%s",
                raw_content[:500],
            )

        return self._fallback_payload(query)

    def _fallback_payload(self, query: str) -> dict:
        return {
            "problems": [
                {
                    "problem": self._fallback_problem(query),
                    "emotions": [self._infer_fallback_emotion(query)],
                }
            ]
        }

    @staticmethod
    def _fallback_problem(query: str) -> str:
        tokens = re.findall(r"\b[\w']+\b", query.strip())
        problem = " ".join(tokens[:8]).strip()
        return problem if len(problem) >= 3 else "generic informational query"

    @staticmethod
    def _infer_fallback_emotion(query: str) -> EmotionLabel:
        q = query.lower()
        if any(word in q for word in {"worried", "anxious", "anxiety", "stress", "stressed"}):
            return "anxiety"
        if any(word in q for word in {"fear", "afraid", "scared"}):
            return "fear"
        if any(word in q for word in {"confused", "confusion", "unclear"}):
            return "confusion"
        if any(word in q for word in {"discipline", "focus", "habit", "routine", "action plan"}):
            return "discipline"
        if any(word in q for word in {"angry", "anger"}):
            return "anger"
        if any(word in q for word in {"sad", "grief", "loss"}):
            return "grief"
        return "none"
