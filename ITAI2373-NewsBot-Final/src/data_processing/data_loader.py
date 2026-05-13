"""
Data loader — downloads and loads the BBC News Archive dataset.

Uses kagglehub for automatic download; falls back to manual CSV
discovery if the expected column layout varies by kagglehub version.
"""

import os
import glob
import pandas as pd

EXPECTED_COLUMNS = {"category", "text"}
CATEGORY_COLUMN_ALIASES = ["category", "Category", "label", "Label", "class"]
TEXT_COLUMN_ALIASES    = ["text", "Text", "article", "Article", "content", "Content"]


def load_bbc_dataset(path: str) -> pd.DataFrame:
    """Load the BBC News Archive from a kagglehub download path.

    Searches `path` recursively for the first CSV file and normalises
    column names to 'category' and 'text'.

    Args:
        path: Root directory returned by kagglehub.dataset_download().

    Returns:
        DataFrame with at minimum 'category' and 'text' columns.

    Raises:
        FileNotFoundError: If no CSV is found under `path`.
        ValueError: If required columns cannot be identified.
    """
    csv_files = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under: {path}")

    # Prefer the largest file if multiple CSVs exist
    csv_path = max(csv_files, key=os.path.getsize)
    df = pd.read_csv(csv_path)

    # Normalise column names
    df = _normalise_columns(df)
    return df


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to 'category' and 'text' regardless of source naming."""
    col_map = {}

    # Find category column
    cat_col = next(
        (c for alias in CATEGORY_COLUMN_ALIASES for c in df.columns if c == alias),
        None
    )
    if cat_col and cat_col != "category":
        col_map[cat_col] = "category"

    # Find text column
    text_col = next(
        (c for alias in TEXT_COLUMN_ALIASES for c in df.columns if c == alias),
        None
    )
    if text_col and text_col != "text":
        col_map[text_col] = "text"

    if col_map:
        df = df.rename(columns=col_map)

    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Could not identify required columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    return df


def load_from_csv(csv_path: str) -> pd.DataFrame:
    """Load directly from a known CSV file path.

    Args:
        csv_path: Absolute path to a BBC-format CSV file.

    Returns:
        Normalised DataFrame with 'category' and 'text' columns.
    """
    df = pd.read_csv(csv_path)
    return _normalise_columns(df)


__all__ = ["load_bbc_dataset", "load_from_csv"]
