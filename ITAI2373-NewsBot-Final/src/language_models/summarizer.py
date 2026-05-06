"""
Abstractive summarization using DistilBART (sshleifer/distilbart-cnn-12-6).
"""

from transformers import pipeline as hf_pipeline

from config.settings import (
    SUMMARIZER_MODEL,
    SUMMARY_MAX_LENGTH,
    SUMMARY_MIN_LENGTH,
    SUMMARIZER_INPUT_TOKEN_LIMIT,
)

_summarizer = None


def _get_summarizer():
    global _summarizer
    if _summarizer is None:
        _summarizer = hf_pipeline(
            'summarization',
            model=SUMMARIZER_MODEL,
            device=-1,
        )
    return _summarizer


def summarize(text: str, max_length: int = SUMMARY_MAX_LENGTH,
              min_length: int = SUMMARY_MIN_LENGTH) -> str:
    """Generate an abstractive summary for a single article.

    Args:
        text: Raw article text.
        max_length: Maximum tokens in generated summary.
        min_length: Minimum tokens in generated summary.

    Returns:
        Summary string, or original text if too short to summarize.
    """
    tokens = str(text).split()
    if len(tokens) < 60:
        return text
    truncated = ' '.join(tokens[:SUMMARIZER_INPUT_TOKEN_LIMIT])
    try:
        result = _get_summarizer()(
            truncated,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
        )
        return result[0]['summary_text'].strip()
    except Exception as exc:
        return f'[Summarization error: {exc}]'


def summarize_with_stats(text: str) -> dict:
    """Summarize and return compression statistics.

    Args:
        text: Raw article text.

    Returns:
        Dict with keys: summary, original_words, summary_words, compression_pct.
    """
    summary = summarize(text)
    orig_words = len(str(text).split())
    summ_words = len(summary.split())
    compression = round((1 - summ_words / max(orig_words, 1)) * 100, 1)
    return {
        'summary': summary,
        'original_words': orig_words,
        'summary_words': summ_words,
        'compression_pct': compression,
    }


def batch_summarize(texts) -> list[dict]:
    """Summarize a list of texts.

    Args:
        texts: Iterable of strings.

    Returns:
        List of dicts from summarize_with_stats().
    """
    return [summarize_with_stats(t) for t in texts]
