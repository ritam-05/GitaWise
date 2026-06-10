"""Public exports for the GitaWise query engine package."""

from __future__ import annotations

from typing import Any

__all__ = ["AdaptiveGitaEngine", "GitaQueryEngine"]


def __getattr__(name: str) -> Any:
    if name == "AdaptiveGitaEngine":
        from .adaptive_engine import AdaptiveGitaEngine

        return AdaptiveGitaEngine
    if name == "GitaQueryEngine":
        from .engine import GitaQueryEngine

        return GitaQueryEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
