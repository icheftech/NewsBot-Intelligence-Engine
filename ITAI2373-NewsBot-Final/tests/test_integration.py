"""
Integration tests — end-to-end pipeline smoke tests.
"""

import pytest
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_processing.text_preprocessor import preprocess, preprocess_corpus
from src.data_processing.feature_extractor import fit_tfidf, fit_count
from src.data_processing.data_validator import validate_dataframe, clean_dataframe
from src.analysis.sentiment_analyzer import score_text, score_corpus
from src.analysis.topic_modeler import TopicModeler
from src.conversation.intent_classifier import classify_intent


MINI_CORPUS = [
    {'category': 'sport',    'text': 'The football team won the championship match after a brilliant performance by the striker who scored the winning goal in extra time.'},
    {'category': 'tech',     'text': 'The technology company released a new smartphone with improved artificial intelligence features and better battery performance.'},
    {'category': 'politics', 'text': 'The government minister announced a new policy on economic spending and fiscal management aimed at reducing the national deficit.'},
    {'category': 'business', 'text': 'Stock market shares rose sharply after the company reported record profits beating analyst expectations for the quarter.'},
    {'category': 'entertainment', 'text': 'The award winning film received critical acclaim at the international festival winning the top prize for best picture.'},
] * 4


@pytest.fixture
def mini_df():
    return pd.DataFrame(MINI_CORPUS)


@pytest.fixture
def pipeline(mini_df):
    """Fitted TF-IDF vectorizer and classifier on mini corpus."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    clean = [preprocess(t) for t in mini_df['text']]
    vec, X = fit_tfidf(clean)
    y = mini_df['category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X_train, y_train)
    return vec, clf, clean, y


class TestEndToEndPipeline:
    def test_preprocess_then_tfidf(self, mini_df):
        clean = preprocess_corpus(mini_df['text'])
        vec, X = fit_tfidf(clean)
        assert X.shape[0] == len(mini_df)

    def test_validate_clean_pipeline(self, mini_df):
        result = validate_dataframe(mini_df)
        assert result['passed']
        cleaned = clean_dataframe(mini_df)
        assert len(cleaned) > 0

    def test_sentiment_on_corpus(self, mini_df):
        scores = score_corpus(mini_df['text'])
        assert len(scores) == len(mini_df)
        assert 'sent_compound' in scores.columns

    def test_topic_modeling_lda(self, mini_df):
        clean = preprocess_corpus(mini_df['text'])
        vec, dtm = fit_count(clean)
        tm = TopicModeler(n_topics=3, method='lda')
        dist = tm.fit_transform(dtm, vec)
        assert dist.shape[0] == len(mini_df)

    def test_topic_modeling_nmf(self, mini_df):
        clean = preprocess_corpus(mini_df['text'])
        vec, dtm = fit_count(clean)
        tm = TopicModeler(n_topics=3, method='nmf')
        dist = tm.fit_transform(dtm, vec)
        assert dist.shape[0] == len(mini_df)

    def test_classifier_predicts_valid_category(self, pipeline):
        vec, clf, _, _ = pipeline
        article = preprocess("The prime minister announced a new budget for economic growth")
        X = vec.transform([article])
        pred = clf.predict(X)[0]
        assert pred in {'business', 'entertainment', 'politics', 'sport', 'tech'}

    def test_intent_classifier_in_pipeline(self):
        assert classify_intent("classify: breaking news from parliament") == 'classify'
        assert classify_intent("search: premier league results") == 'search'
