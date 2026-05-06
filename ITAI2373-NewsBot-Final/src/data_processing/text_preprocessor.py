"""
Text preprocessing pipeline — tokenization, stopword removal, lemmatization.
"""

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

for pkg in ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']:
    nltk.download(pkg, quiet=True)

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words('english'))


def preprocess(text: str, return_tokens: bool = False):
    """Clean, tokenize, and lemmatize a single text string.

    Args:
        text: Raw article text.
        return_tokens: If True return list of tokens; otherwise return joined string.

    Returns:
        List of tokens or whitespace-joined cleaned string.
    """
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [
        _lemmatizer.lemmatize(t)
        for t in tokens
        if t not in _stop_words and len(t) > 2
    ]
    return tokens if return_tokens else ' '.join(tokens)


def preprocess_corpus(texts, return_tokens: bool = False):
    """Apply preprocess() to an iterable of texts.

    Args:
        texts: Iterable of raw strings.
        return_tokens: Passed through to preprocess().

    Returns:
        List of cleaned strings or token lists.
    """
    return [preprocess(t, return_tokens=return_tokens) for t in texts]
