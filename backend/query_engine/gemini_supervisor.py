"""REST-based Gemini LLM Supervisor for routing, intent analysis, and response generation control."""

import json
import logging
import os
import requests
from typing import Any, Dict, List, Optional
from .dialogue_state import DialogueState

LOGGER = logging.getLogger(__name__)


class GeminiSupervisor:
    """Supervisor layer using Gemini 1.5 Flash to analyze incoming turns and direct Groq."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model = "gemini-2.5-flash"
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"

    def supervise(
        self,
        user_query: str,
        dialogue_state: Optional[DialogueState] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Check the user query and return supervisor classification, topic, and generation instruction."""
        if not self.api_key:
            LOGGER.warning("[SUPERVISOR] GEMINI_API_KEY not configured. Falling back to RAG pipeline.")
            return {
                "classification": "TOPIC_SHIFT",
                "reason": "gemini_key_missing_fallback",
                "detected_topic": user_query,
                "instruction_for_groq": None,
            }

        current_topic = getattr(dialogue_state, "active_topic", None) or "None"
        history_str = ""
        if conversation_history:
            history_lines = []
            for turn in conversation_history[-6:]:
                role = str(turn.get("role", "")).upper()
                content = str(turn.get("content", "")).strip()[:150]
                if role in {"USER", "ASSISTANT"}:
                    history_lines.append(f"{role}: {content}")
            history_str = "\n".join(history_lines)
        if not history_str:
            history_str = "No prior history."

        prompt = (
            "You are the Supervisor for a Bhagavad Gita RAG conversation assistant named GitaWise.\n"
            "Your task is to analyze the latest user message in the context of the conversation and determine how it should be handled.\n\n"
            f"CURRENT_TOPIC:\n{current_topic}\n\n"
            f"CONVERSATION_HISTORY:\n{history_str}\n\n"
            f"LATEST_USER_MESSAGE:\n{user_query}\n\n"
            "Classify the user message into one of the following classes:\n"
            '1. "CONTINUATION": The message continues the CURRENT_TOPIC or asks a follow-up/clarification question related to the conversation context.\n'
            '2. "TOPIC_SHIFT": The message shifts to a new, distinct Bhagavad Gita concept or personal problem that requires a fresh verse retrieval (e.g. asking about karma, dharma, fear, attachment, etc. when the current topic is something else).\n'
            '3. "FEEDBACK": The message is feedback (positive or negative), an expression of gratitude (e.g., "thank you", "okay thank you so much"), or simple confirmation/acknowledgement (e.g., "okay", "got it", "makes sense").\n'
            '4. "GREETING": The message is a greeting (e.g., "hi", "hello", "namaste", "hey").\n'
            '5. "OTHER": The message is off-topic chat, gibberish, an query about general topics unrelated to Gita/life struggles, or edge cases.\n\n'
            "Respond ONLY with a valid JSON object in this exact schema:\n"
            "{\n"
            '  "classification": "CONTINUATION" | "TOPIC_SHIFT" | "FEEDBACK" | "GREETING" | "OTHER",\n'
            '  "reason": "explanation of your classification decision",\n'
            '  "detected_topic": "the primary topic name (if CONTINUATION or TOPIC_SHIFT, otherwise null)",\n'
            '  "instruction_for_groq": "detailed generation guidance for Groq (ONLY if FEEDBACK, GREETING, or OTHER. For example: \'Respond warmly acknowledging the gratitude/greeting, keep it concise, and ask if they want to explore a new Gita concept or continue. Do not quote verses.\')"\n'
            "}\n\n"
            "Do not output any introductory or concluding text. Output ONLY the JSON block."
        )

        try:
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"responseMimeType": "application/json"},
            }
            params = {"key": self.api_key}
            
            LOGGER.info("[SUPERVISOR] Calling Gemini API...")
            response = requests.post(self.url, params=params, json=payload, headers=headers, timeout=10)
            
            if response.status_code != 200:
                LOGGER.warning(
                    "[SUPERVISOR] Gemini API request failed status_code=%d response=%s",
                    response.status_code,
                    response.text,
                )
                raise RuntimeError(f"Gemini API returned status code {response.status_code}")

            data = response.json()
            text_response = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            result = json.loads(text_response)
            
            # Ensure correct keys and defaults
            result = {
                "classification": str(result.get("classification", "TOPIC_SHIFT")).upper(),
                "reason": str(result.get("reason", "no_reason_provided")),
                "detected_topic": result.get("detected_topic"),
                "instruction_for_groq": result.get("instruction_for_groq"),
            }
            LOGGER.info(
                "[SUPERVISOR] Classification: %s, Topic: %s",
                result["classification"],
                result["detected_topic"],
            )
            return result

        except Exception as exc:
            LOGGER.warning("[SUPERVISOR] Gemini classification failed: %s. Defaulting to standard flow.", exc)
            return {
                "classification": "TOPIC_SHIFT",
                "reason": f"error:{exc}",
                "detected_topic": user_query,
                "instruction_for_groq": None,
            }
