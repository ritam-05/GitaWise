"""Rebuild the Qdrant index from configured GitaWise chunks."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qdrant_client.http import models

from config import (
    GITA_CHUNKS_CSV,
    GITA_EMBEDDINGS_NPY,
    GITA_METADATA_PKL,
    QDRANT_COLLECTION_NAME,
)
from backend.gita_vector_store.generate_embeddings import (
    build_metadata,
    generate_embeddings,
    load_chunks,
    load_embedding_model,
    save_embeddings,
    save_metadata,
    validate_schema,
)
from backend.gita_vector_store.upload_to_qdrant import (
    build_points,
    create_qdrant_client,
    load_environment,
    upload_points,
)
from backend.query_engine.embedding_model import get_embedding_dimension


def _extract_collection_dimension(collection_info: Any) -> int:
    vectors_config = collection_info.config.params.vectors
    if hasattr(vectors_config, "size"):
        return int(vectors_config.size)
    if isinstance(vectors_config, dict) and vectors_config:
        first_vector = next(iter(vectors_config.values()))
        if hasattr(first_vector, "size"):
            return int(first_vector.size)
    raise RuntimeError("Could not determine Qdrant collection vector dimension.")


def delete_collection_if_exists(client, collection_name: str) -> None:
    try:
        client.get_collection(collection_name)
    except Exception:
        print(f"Collection does not exist: {collection_name}")
        return

    print(f"Deleting existing collection: {collection_name}")
    client.delete_collection(collection_name=collection_name)


def create_collection(client, collection_name: str, embedding_dimension: int) -> None:
    print(f"Creating collection: {collection_name} ({embedding_dimension} dimensions)")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=embedding_dimension,
            distance=models.Distance.COSINE,
        ),
    )


def main() -> None:
    print("Starting Qdrant index rebuild")

    endpoint, api_key = load_environment()
    client = create_qdrant_client(endpoint, api_key)

    model = load_embedding_model()
    embedding_dimension = get_embedding_dimension(model)
    print(f"Validated embedding dimension: {embedding_dimension}")

    print(f"Loading chunks: {GITA_CHUNKS_CSV}")
    df = load_chunks(GITA_CHUNKS_CSV)
    validate_schema(df)
    texts = df["retrieval_text"].astype(str).str.strip().tolist()
    if any(not text for text in texts):
        raise ValueError("Found empty retrieval_text values in gita_chunks.csv.")

    embeddings = generate_embeddings(texts, model)
    if embeddings.shape[1] != embedding_dimension:
        raise RuntimeError(
            f"Generated embedding dimension ({embeddings.shape[1]}) does not match "
            f"model dimension ({embedding_dimension})."
        )

    metadata = build_metadata(df)
    if len(embeddings) != len(metadata):
        raise ValueError(
            f"Embedding count ({len(embeddings)}) does not match metadata count ({len(metadata)})."
        )

    print(f"Saving regenerated embeddings: {GITA_EMBEDDINGS_NPY}")
    save_embeddings(embeddings, GITA_EMBEDDINGS_NPY)
    print(f"Saving regenerated metadata: {GITA_METADATA_PKL}")
    save_metadata(metadata, GITA_METADATA_PKL)

    delete_collection_if_exists(client, QDRANT_COLLECTION_NAME)
    create_collection(client, QDRANT_COLLECTION_NAME, embedding_dimension)

    points = build_points(embeddings, metadata)
    uploaded = upload_points(client, QDRANT_COLLECTION_NAME, points)
    print(f"Uploaded vectors: {uploaded}")

    collection_info = client.get_collection(QDRANT_COLLECTION_NAME)
    collection_dimension = _extract_collection_dimension(collection_info)
    print(f"Qdrant collection dimension: {collection_dimension}")

    if collection_dimension != embedding_dimension:
        raise RuntimeError(
            f"Embedding model dimension ({embedding_dimension}) "
            f"does not match Qdrant collection dimension ({collection_dimension}). "
            "Rebuild the collection or switch embedding models."
        )

    print(
        "Rebuild complete: "
        f"collection={QDRANT_COLLECTION_NAME}, dimension={collection_dimension}, vectors={uploaded}"
    )


if __name__ == "__main__":
    main()
