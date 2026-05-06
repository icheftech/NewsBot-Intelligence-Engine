"""
Topic modeling — LDA and NMF with pyLDAvis visualization support.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation, NMF

from config.settings import (
    N_TOPICS, LDA_MAX_ITER, NMF_MAX_ITER, RANDOM_STATE,
)


class TopicModeler:
    """Unified LDA / NMF topic modeler.

    Args:
        n_topics: Number of latent topics to extract.
        method: 'lda' or 'nmf'.
    """

    def __init__(self, n_topics: int = N_TOPICS, method: str = 'lda'):
        self.n_topics = n_topics
        self.method = method.lower()
        self.model = None
        self.vectorizer = None
        self.dtm = None
        self.vocab = None

    def _build_model(self):
        if self.method == 'lda':
            return LatentDirichletAllocation(
                n_components=self.n_topics,
                random_state=RANDOM_STATE,
                learning_method='online',
                max_iter=LDA_MAX_ITER,
            )
        elif self.method == 'nmf':
            return NMF(
                n_components=self.n_topics,
                random_state=RANDOM_STATE,
                max_iter=NMF_MAX_ITER,
                init='nndsvda',
            )
        else:
            raise ValueError(f"Unknown method '{self.method}'. Choose 'lda' or 'nmf'.")

    def fit_transform(self, dtm, vectorizer):
        """Train the topic model on a document-term matrix.

        Args:
            dtm: Sparse count matrix (n_docs, n_vocab).
            vectorizer: Fitted CountVectorizer that produced dtm.

        Returns:
            topic_dist: (n_docs, n_topics) array of topic weights per document.
        """
        self.vectorizer = vectorizer
        self.dtm = dtm
        self.vocab = vectorizer.get_feature_names_out()
        self.model = self._build_model()
        topic_dist = self.model.fit_transform(dtm)
        return topic_dist

    def get_topic_words(self, topic_id: int, n_words: int = 10) -> list[str]:
        """Return the top words for a given topic index.

        Args:
            topic_id: Zero-based topic index.
            n_words: Number of top words to return.

        Returns:
            List of word strings sorted by weight descending.
        """
        if self.model is None:
            raise RuntimeError("Call fit_transform() first.")
        component = self.model.components_[topic_id]
        top_indices = component.argsort()[::-1][:n_words]
        return [self.vocab[i] for i in top_indices]

    def print_topics(self, n_words: int = 8):
        """Print top words for all topics.

        Args:
            n_words: Words per topic to display.
        """
        if self.model is None:
            raise RuntimeError("Call fit_transform() first.")
        method_label = self.method.upper()
        print(f"\n📌 Top {n_words} words per {method_label} topic:")
        for i in range(self.n_topics):
            words = self.get_topic_words(i, n_words)
            print(f"  Topic {i+1:2d}: {' | '.join(words)}")

    def assign_dominant_topics(self, topic_dist: np.ndarray) -> pd.DataFrame:
        """Add dominant_topic and topic_confidence columns to a DataFrame-ready dict.

        Args:
            topic_dist: (n_docs, n_topics) array from fit_transform.

        Returns:
            DataFrame with dominant_topic (1-indexed) and topic_confidence.
        """
        return pd.DataFrame({
            'dominant_topic': topic_dist.argmax(axis=1) + 1,
            'topic_confidence': topic_dist.max(axis=1),
        })

    def visualize_topics(self):
        """Render pyLDAvis interactive visualization (LDA only).

        Returns:
            pyLDAvis PreparedData object for display in Jupyter.

        Raises:
            RuntimeError: If method is not LDA or model not fitted.
        """
        if self.method != 'lda':
            raise RuntimeError("pyLDAvis visualization is only supported for LDA.")
        if self.model is None:
            raise RuntimeError("Call fit_transform() first.")
        import pyLDAvis
        import pyLDAvis.lda_model
        pyLDAvis.enable_notebook()
        return pyLDAvis.lda_model.prepare(
            self.model, self.dtm, self.vectorizer, sort_topics=False
        )

    def compare_methods(self, dtm, vectorizer) -> dict:
        """Fit both LDA and NMF and return top words for comparison.

        Args:
            dtm: Sparse document-term matrix.
            vectorizer: Fitted CountVectorizer.

        Returns:
            Dict mapping method -> list of topic word lists.
        """
        results = {}
        for method in ('lda', 'nmf'):
            tm = TopicModeler(n_topics=self.n_topics, method=method)
            tm.fit_transform(dtm, vectorizer)
            results[method] = [
                tm.get_topic_words(i) for i in range(self.n_topics)
            ]
        return results
