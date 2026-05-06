"""
News article classification — Logistic Regression with TF-IDF (97.6% accuracy).
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

from config.settings import CATEGORIES, RANDOM_STATE


class NewsClassifier:
    """Wrapper around a TF-IDF + Logistic Regression classification pipeline."""

    def __init__(self, vectorizer=None):
        self.vectorizer = vectorizer
        self.model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        self.is_fitted = False

    def fit(self, clean_texts, labels, vectorizer=None):
        """Fit the classifier.

        Args:
            clean_texts: Preprocessed text strings.
            labels: Category labels aligned with clean_texts.
            vectorizer: Fitted TfidfVectorizer. If None, uses self.vectorizer.

        Returns:
            self
        """
        if vectorizer is not None:
            self.vectorizer = vectorizer
        X = self.vectorizer.transform(clean_texts)
        self.model.fit(X, labels)
        self.is_fitted = True
        return self

    def predict(self, texts):
        """Predict categories for one or more texts.

        Args:
            texts: Single string or list of preprocessed strings.

        Returns:
            Array of predicted category strings.
        """
        if isinstance(texts, str):
            texts = [texts]
        X = self.vectorizer.transform(texts)
        return self.model.predict(X)

    def predict_proba(self, texts):
        """Return probability distribution over categories.

        Args:
            texts: Single string or list of preprocessed strings.

        Returns:
            2-D array (n_samples, n_categories).
        """
        if isinstance(texts, str):
            texts = [texts]
        X = self.vectorizer.transform(texts)
        return self.model.predict_proba(X)

    def evaluate(self, clean_texts, labels):
        """Evaluate on a held-out set.

        Args:
            clean_texts: Preprocessed strings.
            labels: True category labels.

        Returns:
            Dict with 'accuracy' and 'report' keys.
        """
        X = self.vectorizer.transform(clean_texts)
        preds = self.model.predict(X)
        return {
            'accuracy': accuracy_score(labels, preds),
            'report': classification_report(labels, preds, target_names=sorted(set(labels))),
        }

    def train_test_evaluate(self, tfidf_matrix, labels, test_size=0.2):
        """Split, fit, and evaluate in one call.

        Args:
            tfidf_matrix: Full fitted sparse TF-IDF matrix.
            labels: Category labels.
            test_size: Fraction held out for evaluation.

        Returns:
            Dict with 'accuracy', 'report', 'model'.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            tfidf_matrix, labels,
            test_size=test_size, random_state=RANDOM_STATE, stratify=labels,
        )
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        preds = self.model.predict(X_test)
        return {
            'accuracy': accuracy_score(y_test, preds),
            'report': classification_report(y_test, preds, target_names=CATEGORIES),
            'model': self.model,
        }
