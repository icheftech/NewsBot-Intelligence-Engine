"""
Query processor — routes user input through the intent pipeline and returns a response.
"""

from src.conversation.intent_classifier import classify_intent, extract_content
from src.conversation.response_generator import (
    format_classify_response, format_summary_response,
    format_search_response, format_translate_response,
    format_stats_response, format_topics_response, HELP_TEXT,
)


class QueryProcessor:
    """Routes natural language queries to the appropriate NLP backend.

    Args:
        classifier: Fitted NewsClassifier.
        vectorizer: Fitted TfidfVectorizer.
        preprocessor: Callable(str) -> str for preprocessing.
        summarizer_fn: Callable(str) -> dict (summarize_with_stats).
        search_index: SemanticSearchIndex instance.
        translator_fn: Callable(str) -> dict (translate_and_detect).
        sentiment_fn: Callable(str) -> dict (score_text).
        stats_cache: Pre-computed stats dict.
    """

    def __init__(self, classifier=None, vectorizer=None, preprocessor=None,
                 summarizer_fn=None, search_index=None, translator_fn=None,
                 sentiment_fn=None, stats_cache=None):
        self.classifier = classifier
        self.vectorizer = vectorizer
        self.preprocessor = preprocessor
        self.summarizer_fn = summarizer_fn
        self.search_index = search_index
        self.translator_fn = translator_fn
        self.sentiment_fn = sentiment_fn
        self.stats_cache = stats_cache or {}

    def process(self, user_input: str, history=None) -> str:
        """Process a user query and return a formatted response string.

        Args:
            user_input: Raw user message.
            history: Conversation history (unused in current rule-based implementation).

        Returns:
            Formatted markdown response string.
        """
        intent = classify_intent(user_input)
        content = extract_content(user_input, intent)

        if intent == 'classify':
            if len(content) < 20 or content == user_input:
                return 'Please format as: **classify:** [paste your article text here]'
            clean = self.preprocessor(content)
            X = self.vectorizer.transform([clean])
            category = self.classifier.model.predict(X)[0]
            probs = self.classifier.model.predict_proba(X)[0]
            proba_pairs = sorted(zip(self.classifier.model.classes_, probs), key=lambda x: -x[1])
            sentiment = self.sentiment_fn(content)
            return format_classify_response(category, proba_pairs, sentiment)

        elif intent == 'summarize':
            if len(content.split()) < 40 or content == user_input:
                return 'Format: **summarize:** [paste article text] (minimum 40 words)'
            stats = self.summarizer_fn(content)
            return format_summary_response(stats)

        elif intent == 'search':
            query = content if content != user_input else user_input
            if len(query) < 3:
                return 'Format: **search:** [topic or question]'
            results = self.search_index.search(query)
            return format_search_response(query, results)

        elif intent == 'translate':
            if content == user_input:
                return 'Format: **translate:** [non-English text]'
            result = self.translator_fn(content)
            if result.get('error'):
                return f'Translation error: {result["error"]}'
            if result['source_lang'] != 'unknown':
                clean = self.preprocessor(result['translated'])
                X = self.vectorizer.transform([clean])
                result['predicted_category'] = self.classifier.model.predict(X)[0]
                result['sentiment_compound'] = self.sentiment_fn(result['translated'])['sent_compound']
            return format_translate_response(result)

        elif intent == 'stats':
            return format_stats_response(self.stats_cache)

        elif intent == 'topics':
            return format_topics_response(self.stats_cache.get('top_topics', {}))

        elif intent == 'help':
            return HELP_TEXT

        else:
            results = self.search_index.search(user_input, top_k=2)
            resp = 'I found some related articles. Type **help** to see all commands.\n\n'
            for i, r in enumerate(results, 1):
                resp += f'**{i}.** [{r["category"].upper()}] {r["snippet"][:150]}...\n\n'
            return resp
