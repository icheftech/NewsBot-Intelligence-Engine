"""
Content insight generation — key finding extraction from article collections.
"""

from collections import Counter
import re


def extract_key_phrases(text: str, n: int = 5) -> list[str]:
    """Extract the most frequent meaningful phrases (2-3 word n-grams) from text.

    Args:
        text: Preprocessed article text (tokens joined by spaces).
        n: Number of key phrases to return.

    Returns:
        List of phrase strings.
    """
    words = text.lower().split()
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    trigrams = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words) - 2)]
    counts = Counter(bigrams + trigrams)
    return [phrase for phrase, _ in counts.most_common(n)]


def generate_category_insights(df) -> dict:
    """Generate high-level insight strings per category.

    Args:
        df: DataFrame with 'category', 'sent_compound', 'word_count' columns.

    Returns:
        Dict mapping category -> insight string.
    """
    insights = {}
    for cat, group in df.groupby('category'):
        avg_sent = group['sent_compound'].mean() if 'sent_compound' in group else 0
        avg_words = group['word_count'].mean() if 'word_count' in group else 0
        tone = 'positive' if avg_sent >= 0.05 else ('negative' if avg_sent <= -0.05 else 'neutral')
        insights[cat] = (
            f"{cat.title()} articles average {int(avg_words)} words with a {tone} tone "
            f"(mean sentiment: {avg_sent:.3f})."
        )
    return insights
