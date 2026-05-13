"""
Preprocessor — public alias module.

Notebooks and src modules import `preprocess_text` from here.
Delegates to text_preprocessor.preprocess() which handles
tokenization, stopword removal, and lemmatization.
"""

from src.data_processing.text_preprocessor import preprocess, preprocess_corpus


def preprocess_text(text: str) -> str:
    """Clean, tokenize, remove stopwords, and lemmatize a single text string.

    Args:
        text: Raw article text.

    Returns:
        Whitespace-joined cleaned string ready for vectorization.
    """
    return preprocess(text, return_tokens=False)


def preprocess_texts(texts) -> list:
    """Apply preprocess_text() to an iterable of raw strings.

    Args:
        texts: Iterable of raw article strings.

    Returns:
        List of cleaned strings.
    """
    return preprocess_corpus(texts, return_tokens=False)


__all__ = ["preprocess_text", "preprocess_texts"]
