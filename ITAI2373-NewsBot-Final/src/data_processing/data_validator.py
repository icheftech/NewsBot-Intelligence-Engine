"""
Data quality validation for the BBC news DataFrame.
"""

import pandas as pd

REQUIRED_COLUMNS = {'category', 'text'}
VALID_CATEGORIES = {'business', 'entertainment', 'politics', 'sport', 'tech'}
MIN_TEXT_LENGTH = 50


def validate_dataframe(df: pd.DataFrame) -> dict:
    """Run quality checks on the raw DataFrame.

    Args:
        df: DataFrame with at least 'category' and 'text' columns.

    Returns:
        Dict with keys 'passed' (bool), 'issues' (list of str), 'stats' (dict).
    """
    issues = []
    stats = {}

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        issues.append(f"Missing required columns: {missing}")

    null_counts = df[list(REQUIRED_COLUMNS & set(df.columns))].isnull().sum().to_dict()
    stats['null_counts'] = null_counts
    if any(null_counts.values()):
        issues.append(f"Null values detected: {null_counts}")

    if 'category' in df.columns:
        unknown = set(df['category'].unique()) - VALID_CATEGORIES
        if unknown:
            issues.append(f"Unknown categories found: {unknown}")
        stats['category_distribution'] = df['category'].value_counts().to_dict()

    if 'text' in df.columns:
        short_texts = (df['text'].str.len() < MIN_TEXT_LENGTH).sum()
        if short_texts:
            issues.append(f"{short_texts} articles below minimum text length ({MIN_TEXT_LENGTH} chars)")
        stats['avg_text_length'] = df['text'].str.len().mean()

    stats['total_rows'] = len(df)
    return {
        'passed': len(issues) == 0,
        'issues': issues,
        'stats': stats,
    }


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop nulls, filter to valid categories, and normalise column types.

    Args:
        df: Raw DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    df = df.copy()
    if 'category' in df.columns:
        df['category'] = df['category'].str.lower().str.strip()
        df = df[df['category'].isin(VALID_CATEGORIES)]
    if 'text' in df.columns:
        df = df[df['text'].str.len() >= MIN_TEXT_LENGTH]
    return df[list(REQUIRED_COLUMNS)].dropna().reset_index(drop=True)
