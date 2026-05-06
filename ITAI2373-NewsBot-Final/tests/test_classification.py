"""
Unit tests for classification, sentiment analysis, and NER.
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.analysis.sentiment_analyzer import score_text, score_corpus
from src.conversation.intent_classifier import classify_intent, extract_content


class TestSentimentAnalyzer:
    def test_returns_required_keys(self):
        result = score_text("The team won the championship in a thrilling match!")
        for key in ('sent_compound', 'sent_pos', 'sent_neg', 'sent_neu', 'sent_label'):
            assert key in result

    def test_positive_text_classified_positive(self):
        result = score_text("Excellent results! Outstanding performance and amazing achievements.")
        assert result['sent_label'] == 'positive'
        assert result['sent_compound'] > 0

    def test_negative_text_classified_negative(self):
        result = score_text("Terrible failure. Awful results and disastrous consequences.")
        assert result['sent_label'] == 'negative'
        assert result['sent_compound'] < 0

    def test_neutral_text_classified_neutral(self):
        result = score_text("The meeting was held on Tuesday.")
        assert result['sent_label'] == 'neutral'

    def test_corpus_returns_list(self):
        texts = ["Great news!", "Bad news.", "Some news."]
        result = score_corpus(texts)
        assert len(result) == 3

    def test_empty_string_handled(self):
        result = score_text("")
        assert 'sent_compound' in result


class TestIntentClassifier:
    def test_classify_intent(self):
        assert classify_intent("classify: The prime minister announced...") == 'classify'

    def test_summarize_intent(self):
        assert classify_intent("summarize: Long article text here...") == 'summarize'

    def test_search_intent(self):
        assert classify_intent("search: AI and machine learning") == 'search'

    def test_translate_intent(self):
        assert classify_intent("translate: El gobierno anunció...") == 'translate'

    def test_stats_intent(self):
        assert classify_intent("what is the accuracy of the system?") == 'stats'

    def test_topics_intent(self):
        assert classify_intent("show me the topic keywords") == 'topics'

    def test_help_intent(self):
        assert classify_intent("help") == 'help'

    def test_fallback_intent(self):
        assert classify_intent("xyzzy random gibberish input") == 'fallback'

    def test_extract_content_classify(self):
        content = extract_content("classify: The prime minister announced new policies", 'classify')
        assert 'prime minister' in content.lower()

    def test_extract_content_no_prefix(self):
        # Falls back to full query when no command prefix found
        query = "tell me about politics"
        content = extract_content(query, 'classify')
        assert content == query.strip()
