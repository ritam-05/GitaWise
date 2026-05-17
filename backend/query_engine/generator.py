"""Grounded response generation from retrieved Bhagavad Gita contexts."""

from __future__ import annotations

import re
from typing import Optional

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
        conversation_history: list[dict[str, str]] | None = None,
    ) -> GeneratedAnswer:
        """
        Generate grounded answer with optional conversation history for context chaining.
        
        Args:
            user_query: Current user query
            problems: Identified problems/topics
            emotions: Detected emotions
            contexts: Retrieved Gita verses
            warnings: Accumulated warnings
            conversation_history: Previous turns for context (list of {"role": "user"/"assistant", "content": "..."})
        """
        if not contexts:
            raise ValueError("Cannot generate grounded response without retrieved contexts.")

        prompt = render_ground_response_prompt(
            user_query=user_query,
            problems=[{"problem": item.problem} for item in problems],
            emotions=[{"problem": item.problem, "emotion": item.emotion} for item in emotions],
            contexts=[self._context_payload(item) for item in contexts],
            conversation_history=conversation_history,
        )
        answer = self.groq_client.invoke_text(prompt).strip()
        
        # Sanitize answer before citation extraction
        try:
            answer = self._sanitize_answer(answer)
            LOGGER.debug("Answer sanitized successfully")
        except Exception as exc:
            LOGGER.warning("Answer sanitization failed, using raw answer: %s", exc)
            # Fallback: return answer as-is (already stripped)
        
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

    def _sanitize_answer(self, answer: str) -> str:
        """
        Sanitize and normalize the generated answer for quality and safety.
        
        Removes:
        - Malformed markdown (unmatched code blocks, incomplete formatting)
        - Repeated/excessive whitespace and newlines
        - Accidental JSON wrappers or system artifacts
        - Hallucinated metadata
        - Excessive blank lines
        
        Preserves:
        - Valid Bhagavad Gita citations (e.g., "Bhagavad Gita 2.47")
        - Proper philosophical tone and meaning
        - Intentional formatting and emphasis
        
        Args:
            answer: Raw generated answer string from LLM
            
        Returns:
            Cleaned and normalized answer string
            
        Raises:
            ValueError: If sanitization fails critically
        """
        if not isinstance(answer, str):
            raise ValueError(f"Expected string answer, got {type(answer)}")
        
        if not answer:
            return ""
        
        # 1. Strip leading/trailing whitespace
        text = answer.strip()
        
        # 2. Remove accidental JSON wrappers (e.g., {"answer": "..."})
        # Check for common JSON patterns at start/end
        if text.startswith("{") and text.endswith("}"):
            try:
                # Attempt to parse as JSON to extract content
                import json
                try:
                    parsed = json.loads(text)
                    # Extract "answer" or "response" field if present
                    if isinstance(parsed, dict):
                        for key in ["answer", "response", "text", "content", "message"]:
                            if key in parsed and isinstance(parsed[key], str):
                                text = parsed[key].strip()
                                LOGGER.debug("Extracted answer from JSON wrapper (key: %s)", key)
                                break
                except json.JSONDecodeError:
                    # Not valid JSON, proceed with text as-is
                    pass
            except ImportError:
                pass
        
        # 3. Remove markdown code block fencing if malformed
        # But preserve intentional code blocks
        # Remove triple backticks if they're incomplete or orphaned
        backtick_count = text.count("```")
        if backtick_count == 1:
            # Odd number of backticks = incomplete block
            text = text.replace("```", "").strip()
            LOGGER.debug("Removed incomplete code fence markers")
        elif backtick_count > 2 and backtick_count % 2 != 0:
            # Odd number of multiple backticks = malformed
            text = text.replace("```", "").strip()
            LOGGER.debug("Removed malformed code fence markers")
        
        # 4. Normalize excessive whitespace within lines (preserve single spaces)
        # Replace multiple spaces with single space
        text = re.sub(r' {2,}', ' ', text)
        
        # 5. Normalize excessive newlines (preserve paragraph breaks but remove >2 consecutive)
        # Replace 3+ consecutive newlines with 2 (paragraph break)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 6. Remove trailing whitespace from each line (but keep structure)
        lines = text.split('\n')
        lines = [line.rstrip() for line in lines]
        text = '\n'.join(lines)
        
        # 7. Remove common system artifacts/hallucinations
        # Remove common LLM output noise patterns
        artifacts_to_remove = [
            r'^\s*\[.*?system.*?\]\s*$',  # [system: ...] patterns
            r'^\s*<.*?>\s*$',  # XML-like tags on their own line
            r'^\s*\(end of response\)\s*$',  # "(end of response)" markers
            r'^\s*---+\s*$',  # Excessive separator lines
        ]
        
        for pattern in artifacts_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        # 8. Final cleanup: remove leading/trailing whitespace again
        text = text.strip()
        
        # 9. Verify answer is not empty after sanitization
        if not text:
            raise ValueError("Answer became empty after sanitization")
        
        LOGGER.debug("Answer sanitization complete (length: %d chars)", len(text))
        return text

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
        conversation_history: list[dict[str, str]] | None = None,
    ) -> AdaptiveAnswer:
        """
        Generate direct non-RAG response with optional conversation history.
        
        Args:
            user_query: Current user query
            route: Selected route type
            warnings: Accumulated warnings
            fallback_note: Route-specific fallback instruction
            conversation_history: Previous turns for context chaining
        """
        answer = self.groq_client.invoke_text(
            render_direct_response_prompt(
                user_query=user_query,
                route=route,
                fallback_note=fallback_note,
                conversation_history=conversation_history,
            )
        ).strip()
        
        # Sanitize direct response answer as well
        try:
            sanitizer = GroundedResponseGenerator(self.groq_client)
            answer = sanitizer._sanitize_answer(answer)
            LOGGER.debug("Direct response answer sanitized successfully")
        except Exception as exc:
            LOGGER.warning("Direct response sanitization failed, using raw answer: %s", exc)
            # Fallback: return answer as-is (already stripped)
        
        return AdaptiveAnswer(
            original_query=user_query,
            route=route,
            answer=answer,
            cited_verses=[],
            contexts=[],
            warnings=list(warnings or []),
            used_rag=False,
        )
