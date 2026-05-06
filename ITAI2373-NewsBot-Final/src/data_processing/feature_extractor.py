"""
Feature extraction — TF-IDF vectors and count matrices for downstream models.
"""

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from config.settings import (
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE,
    COUNT_VEC_MAX_FEATURES, COUNT_VEC_MIN_DF, COUNT_VEC_MAX_DF,
)


def build_tfidf_vectorizer(**kwargs) -> TfidfVectorizer:
    """Return a configured (unfitted) TF-IDF vectorizer.

    Args:
        **kwargs: Override default settings.

    Returns:
        sklearn TfidfVectorizer instance.
    """
    defaults = dict(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
    )
    defaults.update(kwargs)
    return TfidfVectorizer(**defaults)


def build_count_vectorizer(**kwargs) -> CountVectorizer:
    """Return a configured (unfitted) count vectorizer for topic modeling.

    Args:
        **kwargs: Override default settings.

    Returns:
        sklearn CountVectorizer instance.
    """
    defaults = dict(
        max_features=COUNT_VEC_MAX_FEATURES,
        min_df=COUNT_VEC_MIN_DF,
        max_df=COUNT_VEC_MAX_DF,
    )
    defaults.update(kwargs)
    return CountVectorizer(**defaults)


def fit_tfidf(clean_texts):
    """Fit a TF-IDF vectorizer and transform the corpus.

    Args:
        clean_texts: Iterable of preprocessed strings.

    Returns:
        Tuple of (fitted TfidfVectorizer, sparse matrix X).
    """
    vec = build_tfidf_vectorizer()
    X = vec.fit_transform(clean_texts)
    return vec, X


def fit_count(clean_texts):
    """Fit a count vectorizer and return document-term matrix.

    Args:
        clean_texts: Iterable of preprocessed strings.

    Returns:
        Tuple of (fitted CountVectorizer, sparse DTM).
    """
    vec = build_count_vectorizer()
    dtm = vec.fit_transform(clean_texts)
    return vec, dtm
