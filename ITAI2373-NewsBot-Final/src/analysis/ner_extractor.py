"""
Named entity recognition using spaCy en_core_web_sm.
"""

from collections import defaultdict
import spacy

try:
    _nlp = spacy.load('en_core_web_sm')
except OSError:
    import subprocess
    subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'], check=True)
    _nlp = spacy.load('en_core_web_sm')

ENTITY_TYPES = {'PERSON', 'ORG', 'GPE', 'DATE', 'MONEY'}


def extract_entities(text: str) -> list[dict]:
    """Extract named entities from a single text.

    Args:
        text: Raw article text.

    Returns:
        List of dicts with 'text' and 'label' keys.
    """
    doc = _nlp(str(text)[:100000])
    return [
        {'text': ent.text, 'label': ent.label_}
        for ent in doc.ents
        if ent.label_ in ENTITY_TYPES
    ]


def entity_frequency(texts) -> dict:
    """Count entity occurrences across a corpus.

    Args:
        texts: Iterable of strings.

    Returns:
        Dict mapping entity_label -> Counter of entity texts.
    """
    from collections import Counter
    freq = defaultdict(Counter)
    for text in texts:
        for ent in extract_entities(text):
            freq[ent['label']][ent['text']] += 1
    return dict(freq)


def top_entities(texts, n: int = 10) -> dict:
    """Return the top-n entities per entity type across a corpus.

    Args:
        texts: Iterable of strings.
        n: Number of top entities per type.

    Returns:
        Dict mapping entity_label -> list of (entity_text, count) tuples.
    """
    freq = entity_frequency(texts)
    return {label: counter.most_common(n) for label, counter in freq.items()}
