"""Generate normalized dense embeddings for Bhagavad Gita retrieval chunks."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    GITA_CHUNKS_CSV,
    GITA_EMBEDDINGS_NPY,
    GITA_METADATA_PKL,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_BATCH_SIZE,
    REQUIRED_EMBEDDING_COLUMNS,
    CSV_ENCODING,
)
from backend.query_engine.embedding_model import get_embedding_dimension


def load_chunks(path: Path) -> pd.DataFrame:
    """Load chunked verse data."""
    if not path.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {path}. Run create_chunks.py first."
        )

    return pd.read_csv(path, encoding=CSV_ENCODING).fillna("")


def validate_schema(df: pd.DataFrame) -> None:
    """Ensure every embedding and metadata field exists."""
    missing_columns = [column for column in REQUIRED_EMBEDDING_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "Chunk dataset is missing required columns: "
            f"{missing_columns}. Found columns: {list(df.columns)}"
        )


def load_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    """Load the sentence-transformers model."""
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"Embedding dimension: {get_embedding_dimension(model)}")
    return model


def generate_embeddings(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> np.ndarray:
    """Generate L2-normalized embeddings in batches."""
    if not texts:
        raise ValueError("No retrieval_text values found for embedding.")

    all_embeddings: list[np.ndarray] = []

    for start in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch_texts = texts[start : start + batch_size]
        batch_embeddings = model.encode(
            batch_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        all_embeddings.append(batch_embeddings.astype(np.float32))

    return np.vstack(all_embeddings)


def build_metadata(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Build metadata records for vector-store payloads and reranking."""
    metadata_columns = [
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
    return df[metadata_columns].to_dict(orient="records")


def save_embeddings(embeddings: np.ndarray, path: Path) -> None:
    """Persist embeddings as a NumPy array."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings)


def save_metadata(metadata: list[dict[str, Any]], path: Path) -> None:
    """Persist metadata records using pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(metadata, file)


def main() -> None:
    """Run the embedding generation pipeline."""
    df = load_chunks(GITA_CHUNKS_CSV)
    validate_schema(df)

    texts = df["retrieval_text"].astype(str).str.strip().tolist()
    if any(not text for text in texts):
        raise ValueError("Found empty retrieval_text values in gita_chunks.csv.")

    model = load_embedding_model()
    embedding_dimension = get_embedding_dimension(model)
    embeddings = generate_embeddings(texts, model)
    metadata = build_metadata(df)

    if embeddings.shape[1] != embedding_dimension:
        raise RuntimeError(
            f"Generated embedding dimension ({embeddings.shape[1]}) does not match "
            f"model dimension ({embedding_dimension})."
        )

    if len(embeddings) != len(metadata):
        raise ValueError(
            f"Embedding count ({len(embeddings)}) does not match metadata count ({len(metadata)})."
        )

    save_embeddings(embeddings, GITA_EMBEDDINGS_NPY)
    save_metadata(metadata, GITA_METADATA_PKL)

    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Total vectors generated: {embeddings.shape[0]}")
    print(f"Embeddings saved to: {GITA_EMBEDDINGS_NPY}")
    print(f"Metadata saved to: {GITA_METADATA_PKL}")


if __name__ == "__main__":
    main()
