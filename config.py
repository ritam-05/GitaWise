"""
Centralized Configuration Module for GitaWise Project

This module contains all configuration variables used across the project.
Update values here and all modules will automatically use the new configurations.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# =====================================================
# PROJECT PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_ROOT / "backend"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
DATASETS_DIR = PROJECT_ROOT / "datasets"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# =====================================================
# DATA PATHS
# =====================================================

# Raw and processed data files
CLEAN_GITA_CSV = DATASETS_DIR / "clean_gita.csv"
ENRICHED_GITA_CSV = DATASETS_DIR / "enriched_gita.csv"
BHAGWAD_GITA_CSV = DATASETS_DIR / "Bhagwad_Gita.csv"
BHAGWAD_GITA_JSON = DATASETS_DIR / "Bhagwad_Gita.json"

# Vector store and embedding files
GITA_CHUNKS_CSV = DATASETS_DIR / "gita_chunks.csv"
GITA_EMBEDDINGS_NPY = DATASETS_DIR / "gita_embeddings.npy"
GITA_METADATA_PKL = DATASETS_DIR / "gita_metadata.pkl"

# =====================================================
# EMBEDDING MODEL CONFIGURATION
# =====================================================

# Sentence Transformer model for generating embeddings and runtime queries
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Cross-encoder model for reranking retrieved passages
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# Batch size for embedding generation
EMBEDDING_BATCH_SIZE = 32

# =====================================================
# QDRANT VECTOR STORE CONFIGURATION
# =====================================================

# Collection name in Qdrant
QDRANT_COLLECTION_NAME = "gita_verses"

# Batch size for uploading to Qdrant
QDRANT_UPLOAD_BATCH_SIZE = 64

# Qdrant distance metric (COSINE for normalized embeddings)
QDRANT_DISTANCE_METRIC = "COSINE"

# Qdrant timeout in seconds
QDRANT_TIMEOUT = 60

# =====================================================
# ENVIRONMENT VARIABLES & API CREDENTIALS
# =====================================================

# Load environment variables from .env file
load_dotenv(PROJECT_ROOT / ".env")

# Qdrant credentials
QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY")
QDRANT_ENDPOINT: Optional[str] = os.getenv("QDRANT_ENDPOINT")

# LLM API Keys
GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
SARVAM_API_KEY: Optional[str] = os.getenv("SARVAM_API_KEY")
SUPABASE_URL: Optional[str] = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY: Optional[str] = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY: Optional[str] = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_CACHE_TABLE = os.getenv("SUPABASE_CACHE_TABLE", "cache_entries")

# =====================================================
# DATA SCHEMA CONFIGURATION
# =====================================================

# Required columns in the source dataset
REQUIRED_SOURCE_COLUMNS = [
    "ID",
    "Chapter",
    "Verse",
    "Speaker",
    "Shloka",
    "Transliteration",
    "HinMeaning",
    "EngMeaning",
    "WordMeaning",
    "Interpretation",
    "Topics",
    "EmotionTags",
    "Summary",
]

# Required columns for embeddings and metadata
REQUIRED_EMBEDDING_COLUMNS = [
    "ID",
    "Chapter",
    "Verse",
    "Speaker",
    "retrieval_text",
    "Topics",
    "EmotionTags",
    "Summary",
    "Shloka",
    "EngMeaning",
    "Interpretation",
]

# Metadata columns to include in Qdrant payloads
METADATA_COLUMNS = [
    "ID",
    "Chapter",
    "Verse",
    "Speaker",
    "retrieval_text",
    "Topics",
    "EmotionTags",
    "Summary",
    "Shloka",
    "EngMeaning",
    "Interpretation",
]

# =====================================================
# DATA PROCESSING CONFIGURATION
# =====================================================

# Text encoding for CSV files
CSV_ENCODING = "utf-8-sig"

# =====================================================
# RETRIEVAL TEXT TEMPLATE
# =====================================================

# Template for building retrieval text for embeddings
RETRIEVAL_TEXT_TEMPLATE = """Chapter: {chapter}
Verse: {verse}

Speaker:
{speaker}

Shloka:
{shloka}

Transliteration:
{transliteration}

English Meaning:
{eng_meaning}

Interpretation:
{interpretation}

Topics:
{topics}

Emotion Tags:
{emotion_tags}

Summary:
{summary}"""

# =====================================================
# BACKEND SERVER CONFIGURATION
# =====================================================

# FastAPI server settings
API_HOST = "0.0.0.0"
API_PORT = 8000
API_DEBUG = True
API_RELOAD = True

# =====================================================
# LOGGING CONFIGURATION
# =====================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# =====================================================
# UTILITY FUNCTIONS
# =====================================================


def validate_configuration() -> None:
    """Validate critical configuration values."""
    if not QDRANT_API_KEY:
        raise ValueError("QDRANT_API_KEY is not set in environment variables")
    if not QDRANT_ENDPOINT:
        raise ValueError("QDRANT_ENDPOINT is not set in environment variables")


def get_config_summary() -> dict:
    """Return a summary of active configuration."""
    return {
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "DATASETS_DIR": str(DATASETS_DIR),
        "EMBEDDING_MODEL": EMBEDDING_MODEL_NAME,
        "RERANKER_MODEL": RERANKER_MODEL_NAME,
        "QDRANT_COLLECTION": QDRANT_COLLECTION_NAME,
        "QDRANT_ENDPOINT": QDRANT_ENDPOINT,
        "SUPABASE_URL": SUPABASE_URL,
        "SUPABASE_CACHE_TABLE": SUPABASE_CACHE_TABLE,
    }


if __name__ == "__main__":
    # Print configuration summary when run directly
    import json

    summary = get_config_summary()
    print(json.dumps(summary, indent=2))
