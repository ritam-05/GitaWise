"""Generate normalized dense embeddings for Bhagavad Gita retrieval chunks."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "datasets"
INPUT_PATH = DATASET_DIR / "gita_chunks.csv"
EMBEDDINGS_PATH = DATASET_DIR / "gita_embeddings.npy"
METADATA_PATH = DATASET_DIR / "gita_metadata.pkl"

MODEL_NAME = "BAAI/bge-large-en-v1.5"
BATCH_SIZE = 32

REQUIRED_COLUMNS = [
    "ID",
    "Chapter",
    "Verse",
    "retrieval_text",
    "Topics",
    "EmotionTags",
    "Summary",
    "Shloka",
    "EngMeaning",
    "Interpretation",
]


def load_chunks(path: Path) -> pd.DataFrame:
    """Load chunked verse data."""
    if not path.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {path}. Run create_chunks.py first."
        )

    return pd.read_csv(path, encoding="utf-8-sig").fillna("")


def validate_schema(df: pd.DataFrame) -> None:
    """Ensure every embedding and metadata field exists."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "Chunk dataset is missing required columns: "
            f"{missing_columns}. Found columns: {list(df.columns)}"
        )


def load_embedding_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """Load the sentence-transformers model."""
    print(f"Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)


def generate_embeddings(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int = BATCH_SIZE,
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
    df = load_chunks(INPUT_PATH)
    validate_schema(df)

    texts = df["retrieval_text"].astype(str).str.strip().tolist()
    if any(not text for text in texts):
        raise ValueError("Found empty retrieval_text values in gita_chunks.csv.")

    model = load_embedding_model()
    embeddings = generate_embeddings(texts, model)
    metadata = build_metadata(df)

    if len(embeddings) != len(metadata):
        raise ValueError(
            f"Embedding count ({len(embeddings)}) does not match metadata count ({len(metadata)})."
        )

    save_embeddings(embeddings, EMBEDDINGS_PATH)
    save_metadata(metadata, METADATA_PATH)

    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Total vectors generated: {embeddings.shape[0]}")
    print(f"Embeddings saved to: {EMBEDDINGS_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")


if __name__ == "__main__":
    main()
