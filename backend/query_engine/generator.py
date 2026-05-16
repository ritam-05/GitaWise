"""Grounded response generation from retrieved Bhagavad Gita contexts."""

from __future__ import annotations

import re

from .config import get_logger
from .decomposer import GroqJSONClient
from .models import AdaptiveAnswer, EmotionResult, GeneratedAnswer, Problem, RetrievedVerse
from .prompts import render_direct_response_prompt, render_ground_response_prompt

LOGGER = get_logger(__name__)


class GroundedResponseGenerator:
    """Generate a final grounded answer using only retrieved contexts."""

    def __init__(self, groq_client: GroqJSONClient) -> None:
        self.groq_client = groq_client

    def generate(
        self,
        user_query: str,
        problems: list[Problem],
        emotions: list[EmotionResult],
        contexts: list[RetrievedVerse],
        warnings: list[str] | None = None,
    ) -> GeneratedAnswer:
        if not contexts:
            raise ValueError("Cannot generate grounded response without retrieved contexts.")

        prompt = render_ground_response_prompt(
            user_query=user_query,
            problems=[{"problem": item.problem} for item in problems],
            emotions=[{"problem": item.problem, "emotion": item.emotion} for item in emotions],
            contexts=[self._context_payload(item) for item in contexts],
        )
        answer = self.groq_client.invoke_text(prompt).strip()
        cited_verses = self._extract_citations(answer, contexts)

        LOGGER.info("Generated grounded response with citations: %s", cited_verses)
        return GeneratedAnswer(
            original_query=user_query,
            problems=problems,
            emotions=emotions,
            contexts=contexts,
            answer=answer,
            cited_verses=cited_verses,
            warnings=list(warnings or []),
        )

    @staticmethod
    def _context_payload(context: RetrievedVerse) -> dict[str, object]:
        return {
            "chapter": context.chapter,
            "verse": context.verse,
            "speaker": context.speaker,
            "shloka": context.shloka,
            "translation": context.translation,
            "interpretation": context.interpretation,
            "topics": context.topics,
            "emotion_tags": context.emotion_tags,
            "summary": context.summary,
        }

    @staticmethod
    def _extract_citations(answer: str, contexts: list[RetrievedVerse]) -> list[str]:
        allowed = {f"Bhagavad Gita {item.chapter}.{item.verse}" for item in contexts}
        found = re.findall(r"Bhagavad Gita\s+\d+\.\d+", answer)
        ordered: list[str] = []
        for citation in found:
            if citation in allowed and citation not in ordered:
                ordered.append(citation)
        return ordered


class DirectResponseGenerator:
    """Generate a direct non-RAG answer when retrieval should be bypassed or skipped."""

    def __init__(self, groq_client: GroqJSONClient) -> None:
        self.groq_client = groq_client

    def generate(
        self,
        user_query: str,
        route: str,
        warnings: list[str] | None = None,
        fallback_note: str = "No fallback note.",
    ) -> AdaptiveAnswer:
        answer = self.groq_client.invoke_text(
            render_direct_response_prompt(
                user_query=user_query,
                route=route,
                fallback_note=fallback_note,
            )
        ).strip()
        return AdaptiveAnswer(
            original_query=user_query,
            route=route,
            answer=answer,
            cited_verses=[],
            contexts=[],
            warnings=list(warnings or []),
            used_rag=False,
        )
