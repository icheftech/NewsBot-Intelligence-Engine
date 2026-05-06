"""
Sentence embeddings and semantic search via SentenceTransformers.
"""

import pandas as pd
from sentence_transformers import SentenceTransformer, util

from config.settings import (
    EMBEDDING_MODEL,
    SEMANTIC_INDEX_SIZE,
    SEMANTIC_ARTICLE_TOKEN_LIMIT,
    SEMANTIC_TOP_K,
    RANDOM_STATE,
)

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


class SemanticSearchIndex:
    """Build and query a cosine-similarity semantic search index."""

    def __init__(self, index_size: int = SEMANTIC_INDEX_SIZE):
        self.index_size = index_size
        self.index_df = None
        self.corpus_embeddings = None

    def build(self, df: pd.DataFrame):
        """Sample articles and encode them into an embedding index.

        Args:
            df: DataFrame with a 'text' column (and optionally 'category').

        Returns:
            self
        """
        self.index_df = df.sample(
            n=min(self.index_size, len(df)), random_state=RANDOM_STATE
        ).reset_index(drop=True)

        short_texts = self.index_df['text'].apply(
            lambda t: ' '.join(str(t).split()[:SEMANTIC_ARTICLE_TOKEN_LIMIT])
        )
        self.corpus_embeddings = _get_model().encode(
            short_texts.tolist(),
            convert_to_tensor=True,
            show_progress_bar=True,
        )
        return self

    def search(self, query: str, top_k: int = SEMANTIC_TOP_K) -> list[dict]:
        """Find semantically similar articles for a natural language query.

        Args:
            query: Free-text search query.
            top_k: Number of results to return.

        Returns:
            List of dicts with 'score', 'category', and 'snippet' keys.
        """
        if self.corpus_embeddings is None:
            raise RuntimeError("Call build() before search().")
        q_emb = _get_model().encode(query, convert_to_tensor=True)
        hits = util.semantic_search(q_emb, self.corpus_embeddings, top_k=top_k)[0]
        results = []
        for hit in hits:
            row = self.index_df.iloc[hit['corpus_id']]
            results.append({
                'score': round(hit['score'], 4),
                'category': row.get('category', 'unknown'),
                'snippet': str(row['text'])[:200] + '...',
            })
        return results
