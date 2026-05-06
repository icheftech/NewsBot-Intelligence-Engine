"""
Model evaluation utilities — metrics, cross-validation, comparison tables.
"""

import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np

from config.settings import RANDOM_STATE, CATEGORIES


def full_classification_report(y_true, y_pred, labels=None) -> dict:
    """Return accuracy + full sklearn classification report.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        labels: Optional label order.

    Returns:
        Dict with 'accuracy' and 'report' (string).
    """
    labels = labels or sorted(set(y_true))
    return {
        'accuracy': round(accuracy_score(y_true, y_pred), 4),
        'report': classification_report(y_true, y_pred, target_names=labels),
    }


def cross_validate_classifier(model, X, y, cv: int = 5) -> dict:
    """Run stratified k-fold cross-validation.

    Args:
        model: sklearn-compatible classifier.
        X: Feature matrix.
        y: Labels.
        cv: Number of folds.

    Returns:
        Dict with 'mean_accuracy', 'std', 'scores'.
    """
    cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=cv_obj, scoring='accuracy')
    return {
        'mean_accuracy': round(scores.mean(), 4),
        'std': round(scores.std(), 4),
        'scores': scores.tolist(),
    }


def compare_classifiers(classifiers: dict, X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """Fit and compare multiple classifiers on the same split.

    Args:
        classifiers: Dict mapping name -> unfitted sklearn classifier.
        X_train, X_test: Feature matrices.
        y_train, y_test: Labels.

    Returns:
        DataFrame with classifier names and their accuracy scores.
    """
    rows = []
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        rows.append({'classifier': name, 'accuracy': round(acc, 4)})
    return pd.DataFrame(rows).sort_values('accuracy', ascending=False).reset_index(drop=True)


def confusion_matrix_df(y_true, y_pred, labels=None) -> pd.DataFrame:
    """Return confusion matrix as a labeled DataFrame.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        labels: Optional label order.

    Returns:
        DataFrame indexed and columned by label names.
    """
    labels = labels or sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=labels, columns=labels)
