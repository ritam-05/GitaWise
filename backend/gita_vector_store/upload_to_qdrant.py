"""Upload Bhagavad Gita verse embeddings and payloads to Qdrant."""

from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from tqdm.auto import tqdm

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing dependency 'qdrant-client'. Install it in the Python environment "
        "that runs this script with: python -m pip install qdrant-client"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "datasets"
EMBEDDINGS_PATH = DATASET_DIR / "gita_embeddings.npy"
METADATA_PATH = DATASET_DIR / "gita_metadata.pkl"

COLLECTION_NAME = "gita_verses"
UPLOAD_BATCH_SIZE = 64


def load_environment() -> tuple[str, str]:
    """Load Qdrant credentials from .env."""
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("QDRANT_API_KEY")
    endpoint = os.getenv("QDRANT_ENDPOINT")

    if not api_key:
        raise ValueError("QDRANT_API_KEY is missing from environment variables.")
    if not endpoint:
        raise ValueError("QDRANT_ENDPOINT is missing from environment variables.")

    return endpoint, api_key


def load_embeddings(path: Path) -> np.ndarray:
    """Load vectors from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: {path}. Run generate_embeddings.py first."
        )

    embeddings = np.load(path)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array, got shape {embeddings.shape}.")

    return embeddings.astype(np.float32)


def load_metadata(path: Path) -> list[dict[str, Any]]:
    """Load payload metadata records from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {path}. Run generate_embeddings.py first."
        )

    with path.open("rb") as file:
        metadata = pickle.load(file)

    if not isinstance(metadata, list):
        raise ValueError("Metadata file must contain a list of dictionaries.")

    return metadata


def create_qdrant_client(endpoint: str, api_key: str) -> QdrantClient:
    """Create a Qdrant client for the configured cloud/local endpoint."""
    return QdrantClient(url=endpoint, api_key=api_key, timeout=60)


def recreate_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
) -> None:
    """Create or replace the target collection with cosine distance."""
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )
    print(f"Collection creation success: {collection_name} ({vector_size} dimensions)")


def stable_point_id(metadata: dict[str, Any], fallback_index: int) -> int:
    """Create a deterministic unsigned integer point ID from verse identity."""
    raw_id = metadata.get("ID") or f"{metadata.get('Chapter')}-{metadata.get('Verse')}"
    if not raw_id:
        raw_id = f"row-{fallback_index}"

    digest = hashlib.sha256(str(raw_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def split_tags(value: Any) -> list[str]:
    """Convert comma-separated tag strings into clean payload lists."""
    if value is None:
        return []

    return [tag.strip() for tag in str(value).split(",") if tag.strip()]


def build_payload(metadata: dict[str, Any]) -> dict[str, Any]:
    """Map metadata into the production payload shape."""
    return {
        "id": metadata.get("ID", ""),
        "chapter": int(metadata.get("Chapter", 0)),
        "verse": int(metadata.get("Verse", 0)),
        "topics": split_tags(metadata.get("Topics", "")),
        "emotion_tags": split_tags(metadata.get("EmotionTags", "")),
        "summary": metadata.get("Summary", ""),
        "shloka": metadata.get("Shloka", ""),
        "translation": metadata.get("EngMeaning", ""),
        "interpretation": metadata.get("Interpretation", ""),
        "retrieval_text": metadata.get("retrieval_text", ""),
    }


def build_points(
    embeddings: np.ndarray,
    metadata: list[dict[str, Any]],
) -> list[models.PointStruct]:
    """Build Qdrant point structs from embeddings and metadata."""
    if len(embeddings) != len(metadata):
        raise ValueError(
            f"Embeddings count ({len(embeddings)}) does not match metadata count ({len(metadata)})."
        )

    points: list[models.PointStruct] = []
    seen_ids: set[int] = set()

    for index, (vector, item) in enumerate(zip(embeddings, metadata)):
        point_id = stable_point_id(item, index)
        while point_id in seen_ids:
            point_id += 1

        seen_ids.add(point_id)
        points.append(
            models.PointStruct(
                id=point_id,
                vector=vector.tolist(),
                payload=build_payload(item),
            )
        )

    return points


def upload_points(
    client: QdrantClient,
    collection_name: str,
    points: list[models.PointStruct],
    batch_size: int = UPLOAD_BATCH_SIZE,
) -> int:
    """Upload points to Qdrant in batches."""
    uploaded = 0

    for start in tqdm(range(0, len(points), batch_size), desc="Uploading to Qdrant"):
        batch = points[start : start + batch_size]
        client.upsert(collection_name=collection_name, points=batch, wait=True)
        uploaded += len(batch)

    return uploaded


def main() -> None:
    """Run the Qdrant upload pipeline."""
    try:
        endpoint, api_key = load_environment()
        embeddings = load_embeddings(EMBEDDINGS_PATH)
        metadata = load_metadata(METADATA_PATH)

        vector_size = embeddings.shape[1]
        client = create_qdrant_client(endpoint, api_key)

        recreate_collection(client, COLLECTION_NAME, vector_size)
        points = build_points(embeddings, metadata)
        uploaded = upload_points(client, COLLECTION_NAME, points)

        print(f"Vectors uploaded: {uploaded}")
        print(f"Upload completion status: completed for collection '{COLLECTION_NAME}'")

    except Exception as exc:
        print(f"Upload failed: {exc}")
        raise


if __name__ == "__main__":
    main()
