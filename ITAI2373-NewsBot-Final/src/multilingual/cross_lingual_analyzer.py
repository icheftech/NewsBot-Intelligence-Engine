"""
Cross-language analysis — translate non-English articles then classify and score sentiment.
"""

import pandas as pd

from src.multilingual.translator import translate_and_detect
from src.multilingual.language_detector import detect_language
from src.analysis.sentiment_analyzer import score_text


def analyze_foreign_article(text: str, classifier, vectorizer, preprocessor) -> dict:
    """Full pipeline for a non-English article.

    Detects language → translates → preprocesses → classifies → scores sentiment.

    Args:
        text: Source-language article text.
        classifier: Fitted NewsClassifier instance.
        vectorizer: Fitted TfidfVectorizer instance.
        preprocessor: Callable that preprocesses a string.

    Returns:
        Dict with source_lang, translated, predicted_category, sentiment.
    """
    result = translate_and_detect(text)
    translated = result['translated']
    clean = preprocessor(translated)
    X = vectorizer.transform([clean])
    category = classifier.model.predict(X)[0]
    sentiment = score_text(translated)
    return {
        'source_lang': result['source_lang'],
        'translated': translated[:300],
        'predicted_category': category,
        'sentiment_compound': sentiment['sent_compound'],
        'sentiment_label': sentiment['sent_label'],
        'error': result.get('error'),
    }


def batch_cross_lingual_analysis(texts, classifier, vectorizer, preprocessor) -> pd.DataFrame:
    """Run cross-lingual analysis on a batch of texts.

    Args:
        texts: Iterable of possibly non-English strings.
        classifier: Fitted NewsClassifier.
        vectorizer: Fitted TfidfVectorizer.
        preprocessor: Preprocessing callable.

    Returns:
        DataFrame with one row per article.
    """
    rows = []
    for text in texts:
        lang = detect_language(text)
        if lang != 'en':
            rows.append(analyze_foreign_article(text, classifier, vectorizer, preprocessor))
    return pd.DataFrame(rows)
