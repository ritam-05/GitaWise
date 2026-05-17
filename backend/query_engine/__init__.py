"""Public exports for the GitaWise query engine package."""

import logging
logger = logging.getLogger(__name__)
logger.info("[PACKAGE] backend.query_engine __init__ starting...")

logger.info("[PACKAGE] Importing adaptive_engine...")
from .adaptive_engine import AdaptiveGitaEngine
logger.info("[PACKAGE] ✓ AdaptiveGitaEngine imported")

logger.info("[PACKAGE] Importing engine...")
from .engine import GitaQueryEngine
logger.info("[PACKAGE] ✓ GitaQueryEngine imported")

__all__ = ["AdaptiveGitaEngine", "GitaQueryEngine"]
logger.info("[PACKAGE] ✓ query_engine package fully initialized")
