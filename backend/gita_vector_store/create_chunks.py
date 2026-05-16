"""Create one semantic retrieval chunk per Bhagavad Gita verse.

The chunk text is intentionally structured for downstream dense retrieval,
reranking, citation display, and answer generation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "datasets"
INPUT_PATH = DATASET_DIR / "clean_gita.csv"
OUTPUT_PATH = DATASET_DIR / "gita_chunks.csv"

REQUIRED_COLUMNS = [
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


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the cleaned dataset with UTF-8-safe handling."""
    if not path.exists():
        raise FileNotFoundError(f"Input dataset not found: {path}")

    return pd.read_csv(path, encoding="utf-8-sig").fillna("")


def validate_schema(df: pd.DataFrame) -> None:
    """Validate that all required source fields are present."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "Input dataset is missing required columns: "
            f"{missing_columns}. Found columns: {list(df.columns)}"
        )


def build_retrieval_text(row: pd.Series) -> str:
    """Build the exact retrieval text used for embedding."""
    return (
        f"Chapter: {row['Chapter']}\n"
        f"Verse: {row['Verse']}\n\n"
        f"Speaker:\n{row['Speaker']}\n\n"
        f"Shloka:\n{row['Shloka']}\n\n"
        f"Transliteration:\n{row['Transliteration']}\n\n"
        f"English Meaning:\n{row['EngMeaning']}\n\n"
        f"Interpretation:\n{row['Interpretation']}\n\n"
        f"Topics:\n{row['Topics']}\n\n"
        f"Emotion Tags:\n{row['EmotionTags']}\n\n"
        f"Summary:\n{row['Summary']}"
    ).strip()


def create_chunks(df: pd.DataFrame) -> pd.DataFrame:
    """Create one semantic chunk per verse."""
    chunk_df = df.copy()
    chunk_df["retrieval_text"] = chunk_df.apply(build_retrieval_text, axis=1)

    empty_chunks = chunk_df["retrieval_text"].astype(str).str.strip().eq("").sum()
    if empty_chunks:
        raise ValueError(f"Found {empty_chunks} empty retrieval_text chunks.")

    return chunk_df


def save_chunks(df: pd.DataFrame, path: Path) -> None:
    """Save chunks using UTF-8 with BOM for spreadsheet compatibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    """Run the chunk creation pipeline."""
    df = load_dataset(INPUT_PATH)
    validate_schema(df)

    chunks = create_chunks(df)
    save_chunks(chunks, OUTPUT_PATH)

    print(f"Total verses processed: {len(chunks)}")
    print(f"Output file location: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
