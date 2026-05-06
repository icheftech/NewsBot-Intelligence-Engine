"""
Rule-based intent classifier for the NewsBot conversational interface.
"""

INTENT_KEYWORDS = {
    'classify':   ['classify', 'what category', 'what type', 'predict'],
    'summarize':  ['summarize', 'summarise', 'summary', 'tldr', 'tl;dr'],
    'search':     ['search', 'find articles', 'find news', 'related to', 'find:'],
    'translate':  ['translate', 'in spanish', 'in french', 'non-english', 'foreign'],
    'stats':      ['stat', 'how many', 'accuracy', 'performance', 'distribution', 'count', 'breakdown'],
    'topics':     ['topic', 'keyword', 'theme', 'key word'],
    'help':       ['help', 'what can you', 'commands', 'how do i', 'guide'],
}


def classify_intent(query: str) -> str:
    """Determine the user's intent from a natural language query.

    Args:
        query: Raw user input string.

    Returns:
        Intent string: one of 'classify', 'summarize', 'search', 'translate',
        'stats', 'topics', 'help', or 'fallback'.
    """
    q = query.lower().strip()
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return intent
    return 'fallback'


def extract_content(query: str, intent: str) -> str:
    """Extract the content portion of a query after the command keyword.

    Args:
        query: Raw user input.
        intent: Detected intent string.

    Returns:
        Content substring after the command prefix, or full query if not found.
    """
    q_lower = query.lower()
    prefixes = {
        'classify':  ['classify:', 'classify this:', 'classify this article:', 'classify the following:'],
        'summarize': ['summarize:', 'summarise:', 'summary:', 'tldr:', 'tl;dr:'],
        'search':    ['search:', 'find articles about:', 'find:', 'search for:'],
        'translate': ['translate:'],
    }
    for prefix in prefixes.get(intent, []):
        if prefix in q_lower:
            idx = q_lower.index(prefix) + len(prefix)
            return query[idx:].strip()
    return query.strip()
