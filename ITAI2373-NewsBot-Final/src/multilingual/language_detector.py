"""
Language detection using langdetect with reproducible seeding.
"""

from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

from config.settings import LANGDETECT_SEED, LANG_DETECT_TEXT_LIMIT

DetectorFactory.seed = LANGDETECT_SEED


def detect_language(text: str) -> str:
    """Detect the ISO 639-1 language code of a text.

    Args:
        text: Raw text (first LANG_DETECT_TEXT_LIMIT chars are used).

    Returns:
        Language code string (e.g. 'en', 'es') or 'unknown' on failure.
    """
    try:
        return detect(str(text)[:LANG_DETECT_TEXT_LIMIT])
    except LangDetectException:
        return 'unknown'


def detect_corpus_languages(texts) -> list[str]:
    """Detect languages for a list of texts.

    Args:
        texts: Iterable of strings.

    Returns:
        List of language code strings.
    """
    return [detect_language(t) for t in texts]


def language_distribution(texts) -> dict:
    """Count detected languages across a corpus.

    Args:
        texts: Iterable of strings.

    Returns:
        Dict mapping language_code -> count, sorted by frequency.
    """
    from collections import Counter
    codes = detect_corpus_languages(texts)
    return dict(Counter(codes).most_common())
