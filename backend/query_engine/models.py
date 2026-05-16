"""Typed data models for the GitaWise query engine."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

EmotionLabel = Literal[
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
]

RouteLabel = Literal[
    "generic_chat",
    "philosophical_guidance",
    "emotion_guidance",
    "gita_rag",
]


class Problem(BaseModel):
    """A distinct philosophical or emotional concern extracted from a query."""

    problem: str = Field(..., min_length=3, description="A concise concern statement.")


class EmotionResult(BaseModel):
    """The dominant emotion linked to a problem."""

    problem: str = Field(..., min_length=3)
    emotion: EmotionLabel


class EmotionNormalizationResult(BaseModel):
    """Canonical emotion mapping for a raw or user-supplied emotion."""

    input_emotion: str = Field(..., min_length=1)
    mapped_emotions: list[EmotionLabel] = Field(default_factory=list, min_length=1, max_length=3)


class RouteResult(BaseModel):
    """Selected response pipeline for a user query."""

    route: RouteLabel


class RetrievalQuery(BaseModel):
    """A semantically optimized search query for dense retrieval."""

    problem: str = Field(..., min_length=3)
    emotion: EmotionLabel
    query: str = Field(..., min_length=3)


class RetrievedVerse(BaseModel):
    """A verse candidate returned from retrieval or reranking."""

    id: str | None = None
    chapter: int
    verse: int
    speaker: str = ""
    shloka: str = ""
    translation: str = ""
    interpretation: str = ""
    topics: list[str] = Field(default_factory=list)
    emotion_tags: list[str] = Field(default_factory=list)
    summary: str = ""
    retrieval_text: str = ""
    score: float = 0.0
    retrieval_score: float | None = None
    rerank_score: float | None = None
    matched_problems: list[str] = Field(default_factory=list)
    matched_emotions: list[EmotionLabel] = Field(default_factory=list)
    matched_queries: list[str] = Field(default_factory=list)

    @field_validator("topics", "emotion_tags", mode="before")
    @classmethod
    def _ensure_string_list(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return [item.strip() for item in str(value).split(",") if item.strip()]

    @property
    def verse_key(self) -> tuple[int, int]:
        return self.chapter, self.verse


class EngineResponse(BaseModel):
    """Final structured output for the downstream generation layer."""

    original_query: str
    problems: list[Problem]
    emotions: list[EmotionResult]
    retrieval_queries: list[RetrievalQuery]
    contexts: list[RetrievedVerse]
    warnings: list[str] = Field(default_factory=list)


class GeneratedAnswer(BaseModel):
    """Final grounded philosophical answer generated from retrieved context."""

    original_query: str
    problems: list[Problem]
    emotions: list[EmotionResult]
    contexts: list[RetrievedVerse]
    answer: str = Field(..., min_length=1)
    cited_verses: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class AdaptiveAnswer(BaseModel):
    """Final route-aware answer produced by the adaptive orchestration layer."""

    original_query: str
    route: RouteLabel
    answer: str = Field(..., min_length=1)
    cited_verses: list[str] = Field(default_factory=list)
    contexts: list[RetrievedVerse] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    used_rag: bool = False
