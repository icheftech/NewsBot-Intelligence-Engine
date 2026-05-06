"""
Multi-dimension VADER sentiment analysis.
"""

import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon', quiet=True)

from config.settings import VADER_POSITIVE_THRESHOLD, VADER_NEGATIVE_THRESHOLD

_sid = SentimentIntensityAnalyzer()


def score_text(text: str) -> dict:
    """Return VADER compound, positive, negative, neutral scores and label.

    Args:
        text: Raw article text.

    Returns:
        Dict with keys: compound, pos, neg, neu, label.
    """
    scores = _sid.polarity_scores(str(text))
    if scores['compound'] >= VADER_POSITIVE_THRESHOLD:
        label = 'positive'
    elif scores['compound'] <= VADER_NEGATIVE_THRESHOLD:
        label = 'negative'
    else:
        label = 'neutral'
    return {
        'sent_compound': scores['compound'],
        'sent_pos': scores['pos'],
        'sent_neg': scores['neg'],
        'sent_neu': scores['neu'],
        'sent_label': label,
    }


def score_corpus(texts) -> pd.DataFrame:
    """Apply score_text() to an iterable and return a DataFrame.

    Args:
        texts: Iterable of strings.

    Returns:
        DataFrame with sentiment columns.
    """
    return pd.DataFrame([score_text(t) for t in texts])


def sentiment_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean sentiment scores grouped by category.

    Args:
        df: DataFrame with 'category' and sentiment columns.

    Returns:
        DataFrame indexed by category with mean sentiment scores.
    """
    sent_cols = ['sent_compound', 'sent_pos', 'sent_neg', 'sent_neu']
    available = [c for c in sent_cols if c in df.columns]
    return df.groupby('category')[available].mean().round(3)
