"""Prompt templates for routing, decomposition, emotion detection, normalization, query building, and generation."""

from __future__ import annotations

import json
import re

from langchain_core.prompts import ChatPromptTemplate

from .models import EmotionLabel

ALLOWED_EMOTIONS: tuple[EmotionLabel, ...] = (
    "none",
    "fear",
    "confusion",
    "grief",
    "anger",
    "anxiety",
    "attachment",
    "doubt",
    "guilt",
    "loneliness",
    "hopelessness",
    "restlessness",
    "peace",
    "courage",
    "surrender",
    "clarity",
    "discipline",
    "frustration",
)

# Compact static prompt blocks injected into both generation prompts.
USER_STYLE_PRIORITY = (
    "FORMAT: Follow the user's requested format exactly "
    "(bullets/table/numbered/paragraphs/short/detailed/step-by-step/Hindi/Sanskrit/transliteration). "
    "Apply the requested format to every main idea. Current query only; ignore prior word counts unless repeated. "
    "If bullets are requested, use bullets from start to finish. Default: prose, 150-200 words."
)

GITA_MENTOR_GUIDE = (
    "METHOD: (1) Anchor in the retrieved verse; cite chapter.verse only if present in context, never fabricate. "
    "(2) Translate the teaching into the user's exact situation; be specific, practical, and not generic. "
    "(3) Explain abstract Gita concepts in plain behavior: what to do, what to stop doing, and how to think. "
    "(4) End with one concrete action the user can take today. "
    "AVOID: therapy-speak, toxic positivity, hollow Gita references, fake Sanskrit, and verse-listing without synthesis."
)

COMBINED_ANALYSIS_PROMPT = ChatPromptTemplate.from_template(
    "Identify 1-3 problems and emotions from the query.\n"
    "Allowed emotions: {allowed_emotions}\n"
    "Return JSON only:\n"
    '{{"problems":[{{"problem":"<10 words","emotions":["anxiety"]}}]}}\n'
    "Rules: distinct problems; every emotion must be exactly one allowed emotion; max 3 per problem; "
    'use "none" only for informational queries with no emotional state.\n\n'
    "Query: {query}"
)

EMOTION_NORMALIZATION_PROMPT = ChatPromptTemplate.from_template(
    "Map the emotion to 1-3 canonical emotions from the allowed list.\n"
    "Allowed: {allowed_emotions}\n"
    'Return JSON only: {{"mapped_emotions":["anxiety"]}}\n'
    'Rules: use only exact allowed labels; prefer 1; use "none" only if no listed emotion fits; no explanation.\n\n'
    "Input: {emotion}"
)

GROUND_RESPONSE_PROMPT = ChatPromptTemplate.from_template(
    "You are GitaWise, a Bhagavad Gita mentor. The retrieved verses are your evidence.\n"
    "{gita_mentor_guide}\n"
    "{user_style_priority}\n"
    "LENGTH: {word_count_instruction}\n\n"
    "FORMAT INSTRUCTION: Write in flowing, natural paragraphs unless the user explicitly requests a specific format (e.g., bullet points, numbered list, table).\n"
    "If the user has NOT requested a special format, write naturally as if having a thoughtful conversation.\n\n"
    "Structure (integrated naturally into paragraphs, not as labeled sections):\n"
    "1. Name the real tension briefly and acknowledge it.\n"
    "2. Explain what the retrieved verses teach in plain modern language.\n"
    "3. Bridge the teaching into the user's specific situation with practical insight.\n"
    "4. Give one or more concrete next steps they can take today.\n"
    "5. Close with one calm, grounding line.\n\n"
    "{conversation_context}"
    "Query: {user_query}\n"
    "Problems: {problems_json}\n"
    "Emotions: {emotions_json}\n"
    "Gita context: {contexts_json}"
)

DIRECT_RESPONSE_PROMPT = ChatPromptTemplate.from_template(
    "You are GitaWise, a Bhagavad Gita mentor. Retrieved verses are unavailable or too weak.\n"
    "{gita_mentor_guide}\n"
    "{user_style_priority}\n"
    "LENGTH: {word_count_instruction}\n"
    "Route: {route} | Note: {fallback_note}\n"
    "Draw on known Gita concepts such as karma yoga, svadharma, nishkama karma, steadiness, and self-mastery, "
    "but do not cite a specific chapter.verse unless it is available in context. "
    "Still follow the mentor method: tension -> teaching -> concrete step.\n\n"
    "{conversation_context}"
    "Query: {user_query}"
)

TOPIC_DECISION_PROMPT = ChatPromptTemplate.from_template(
    "You are a conversation routing engine for a Bhagavad Gita RAG system.\n\n"
    "Your job is to determine whether the latest user message can be answered using the CURRENT conversation topic "
    "and cached retrieval context, or whether a NEW retrieval is required.\n\n"
    "CURRENT_TOPIC:\n"
    "{current_topic}\n\n"
    "CURRENT_CONTEXT_SUMMARY:\n"
    "{context_summary}\n\n"
    "USER_MESSAGE:\n"
    "{user_message}\n\n"
    "Instructions:\n\n"
    "Analyze the user's latest message.\n\n"
    "Return CONTINUE_TOPIC if:\n"
    "- The user is asking for clarification.\n"
    "- The user is asking for examples.\n"
    "- The user is asking for practical applications.\n"
    "- The user is asking follow-up questions.\n"
    "- The user is asking about causes, effects, benefits, drawbacks, comparisons, deeper explanations, or real-life implications of the current topic.\n"
    "- The user uses pronouns such as \"it\", \"this\", \"that\", \"these teachings\", etc. referring to the current topic.\n"
    "- The question can reasonably be answered using the existing retrieved verses and context.\n\n"
    "Return NEW_TOPIC if:\n"
    "- The user introduces a different virtue, emotion, concept, life problem, philosophy, teaching, or subject.\n"
    "- The answer would require retrieving different Bhagavad Gita verses than the current context.\n"
    "- The user shifts from one primary concept to another.\n\n"
    "Examples:\n\n"
    "Current Topic: confidence\n"
    "User: How do I build it?\n"
    "-> CONTINUE_TOPIC\n\n"
    "Current Topic: confidence\n"
    "User: Why do people lose confidence?\n"
    "-> CONTINUE_TOPIC\n\n"
    "Current Topic: confidence\n"
    "User: Can confidence become arrogance?\n"
    "-> CONTINUE_TOPIC\n\n"
    "Current Topic: confidence\n"
    "User: How would Krishna apply this at work?\n"
    "-> CONTINUE_TOPIC\n\n"
    "Current Topic: confidence\n"
    "User: What is patience?\n"
    "-> NEW_TOPIC\n\n"
    "Current Topic: confidence\n"
    "User: What is attachment?\n"
    "-> NEW_TOPIC\n\n"
    "Current Topic: confidence\n"
    "User: Explain fear.\n"
    "-> NEW_TOPIC\n\n"
    "Current Topic: uncertainty\n"
    "User: Explain uncertainty in 5 bullets.\n"
    "-> CONTINUE_TOPIC\n\n"
    "Current Topic: uncertainty\n"
    "User: Give me an example.\n"
    "-> CONTINUE_TOPIC\n\n"
    "Current Topic: uncertainty\n"
    "User: What is confidence?\n"
    "-> NEW_TOPIC\n\n"
    "Output ONLY valid JSON:\n\n"
    "{{\n"
    "  \"decision\": \"CONTINUE_TOPIC\" | \"NEW_TOPIC\",\n"
    "  \"reason\": \"<one sentence>\",\n"
    "  \"detected_topic\": \"<topic name>\"\n"
    "}}\n\n"
    "Do not output anything except JSON."
)

TOPIC_SHIFT_DETECTION_PROMPT = ChatPromptTemplate.from_template(
    "You are a topic-shift detector for a Bhagavad Gita RAG conversation.\n\n"
    "Your only job is to decide whether the latest user message continues the CURRENT_TOPIC "
    "or starts a different primary topic that requires fresh verse retrieval.\n\n"
    "CURRENT_TOPIC:\n"
    "{current_topic}\n\n"
    "CACHED_CONTEXT_SUMMARY:\n"
    "{context_summary}\n\n"
    "LATEST_USER_MESSAGE:\n"
    "{user_message}\n\n"
    "Decision rules:\n"
    "- Return CONTINUATION when the message asks for clarification, examples, bullets, summary, practical application, causes, effects, comparison, deeper explanation, or uses pronouns like it/this/that referring to CURRENT_TOPIC.\n"
    "- Return TOPIC_SHIFT when the message introduces a different main concept, virtue, emotion, life problem, philosophical idea, or subject that would need different Bhagavad Gita verses.\n"
    "- Do not assume all spiritual concepts are the same topic. Confidence, uncertainty, patience, fear, attachment, faith, courage, ego, duty, and karma are distinct primary topics unless the user is explicitly comparing them to CURRENT_TOPIC.\n"
    "- If the latest message is a standalone question about a different concept, return TOPIC_SHIFT even if it is philosophically related.\n\n"
    "Return only valid JSON in this exact shape:\n"
    "{{\n"
    "  \"decision\": \"CONTINUATION\" | \"TOPIC_SHIFT\",\n"
    "  \"reason\": \"<one sentence>\",\n"
    "  \"detected_topic\": \"<current topic if continuation, new topic if topic shift>\"\n"
    "}}\n"
)


def _render_word_count_instruction(user_query: str) -> str:
    """Return response-length guidance based on explicit user word-count requests."""
    match = re.search(
        r"\b(?:in|within|around|about|approx(?:imately)?)?\s*(\d{2,5})\s*words?\b",
        user_query,
        flags=re.IGNORECASE,
    )
    if not match:
        return "Default: 150-200 words unless the user specified another length."

    target_words = int(match.group(1))
    lower_bound = max(1, target_words - 20)
    upper_bound = target_words + 20
    return f"Target: {target_words} words, acceptable range {lower_bound}-{upper_bound}."


def _sanitize_conversation_content(content: str) -> str:
    """Remove stale length-control instructions from previous-turn context."""
    return re.sub(
        r"\b(?:in|within|around|about|approx(?:imately)?)?\s*\d{2,5}\s*words?\b",
        "[previous word-count request omitted]",
        content,
        flags=re.IGNORECASE,
    ).strip()


def _build_conversation_context(
    conversation_history: list[dict[str, str]] | None,
) -> str:
    """Build a compact conversation summary for continuity without stale style leakage."""
    if not conversation_history:
        return ""

    context_lines = ["Prior context:"]
    for turn in conversation_history[-4:]:
        role = turn.get("role", "unknown").upper()
        content = _sanitize_conversation_content(turn.get("content", "").strip())
        if content:
            context_lines.append(f"{role}: {content[:150]}")

    if len(context_lines) == 1:
        return ""
    return "\n".join(context_lines) + "\n\n"


def render_combined_analysis_prompt(query: str) -> str:
    """Render the unified decomposition + emotion detection prompt."""
    return COMBINED_ANALYSIS_PROMPT.format_messages(
        query=query,
        allowed_emotions=", ".join(ALLOWED_EMOTIONS),
    )[0].content


def render_emotion_normalization_prompt(emotion: str) -> str:
    """Render the emotion normalization prompt."""
    return EMOTION_NORMALIZATION_PROMPT.format_messages(
        allowed_emotions=json.dumps(list(ALLOWED_EMOTIONS), ensure_ascii=True),
        emotion=emotion,
    )[0].content


def render_ground_response_prompt(
    user_query: str,
    problems: list[dict[str, str]],
    emotions: list[dict[str, str]],
    contexts: list[dict[str, object]],
    conversation_history: list[dict[str, str]] | None = None,
) -> str:
    """Render grounded response prompt with optional conversation context."""
    return GROUND_RESPONSE_PROMPT.format_messages(
        user_query=user_query,
        problems_json=json.dumps(problems, ensure_ascii=True),
        emotions_json=json.dumps(emotions, ensure_ascii=True),
        contexts_json=json.dumps(contexts, ensure_ascii=True),
        conversation_context=_build_conversation_context(conversation_history),
        word_count_instruction=_render_word_count_instruction(user_query),
        user_style_priority=USER_STYLE_PRIORITY,
        gita_mentor_guide=GITA_MENTOR_GUIDE,
    )[0].content


def render_direct_response_prompt(
    user_query: str,
    route: str,
    fallback_note: str,
    conversation_history: list[dict[str, str]] | None = None,
) -> str:
    """Render direct response prompt with optional conversation context."""
    return DIRECT_RESPONSE_PROMPT.format_messages(
        user_query=user_query,
        route=route,
        fallback_note=fallback_note,
        conversation_context=_build_conversation_context(conversation_history),
        word_count_instruction=_render_word_count_instruction(user_query),
        user_style_priority=USER_STYLE_PRIORITY,
        gita_mentor_guide=GITA_MENTOR_GUIDE,
    )[0].content


def render_topic_decision_prompt(
    current_topic: str,
    context_summary: str,
    user_message: str,
) -> str:
    """Render the cached-context routing prompt."""
    return TOPIC_DECISION_PROMPT.format_messages(
        current_topic=current_topic or "None",
        context_summary=context_summary or "No cached context available.",
        user_message=user_message,
    )[0].content


def render_topic_shift_detection_prompt(
    current_topic: str,
    context_summary: str,
    user_message: str,
) -> str:
    """Render the LLM-only topic-shift detector prompt."""
    return TOPIC_SHIFT_DETECTION_PROMPT.format_messages(
        current_topic=current_topic or "None",
        context_summary=context_summary or "No cached context available.",
        user_message=user_message,
    )[0].content
