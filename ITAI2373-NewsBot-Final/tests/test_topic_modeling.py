"""
Unit tests for TopicModeler (LDA and NMF).
"""

import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sklearn.feature_extraction.text import CountVectorizer
from src.analysis.topic_modeler import TopicModeler


SAMPLE_DOCS = [
    "government minister policy parliament election vote",
    "football team player match season goal win",
    "technology software computer internet data digital",
    "market company share profit growth economy bank",
    "film music award star show album actor director",
    "prime minister budget spending fiscal policy tax",
    "premier league transfer club manager striker",
    "smartphone mobile app software update release",
    "stock market shares price trading investment",
    "celebrity award show entertainment music chart",
] * 5


@pytest.fixture
def fitted_topic_modeler():
    vec = CountVectorizer(max_features=100, min_df=1)
    dtm = vec.fit_transform(SAMPLE_DOCS)
    tm = TopicModeler(n_topics=5, method='lda')
    tm.fit_transform(dtm, vec)
    return tm, dtm, vec


class TestTopicModelerLDA:
    def test_fit_returns_distribution(self, fitted_topic_modeler):
        tm, dtm, vec = fitted_topic_modeler
        dist = tm.model.transform(dtm)
        assert dist.shape == (len(SAMPLE_DOCS), 5)

    def test_get_topic_words_returns_list(self, fitted_topic_modeler):
        tm, _, _ = fitted_topic_modeler
        words = tm.get_topic_words(0, n_words=5)
        assert isinstance(words, list)
        assert len(words) == 5

    def test_dominant_topics_shape(self, fitted_topic_modeler):
        tm, dtm, _ = fitted_topic_modeler
        dist = tm.model.transform(dtm)
        result = tm.assign_dominant_topics(dist)
        assert len(result) == len(SAMPLE_DOCS)
        assert 'dominant_topic' in result.columns
        assert 'topic_confidence' in result.columns

    def test_topic_words_are_strings(self, fitted_topic_modeler):
        tm, _, _ = fitted_topic_modeler
        words = tm.get_topic_words(0)
        assert all(isinstance(w, str) for w in words)

    def test_raises_if_not_fitted(self):
        tm = TopicModeler(n_topics=5, method='lda')
        with pytest.raises(RuntimeError):
            tm.get_topic_words(0)


class TestTopicModelerNMF:
    def test_nmf_fit_and_topics(self):
        vec = CountVectorizer(max_features=100, min_df=1)
        dtm = vec.fit_transform(SAMPLE_DOCS)
        tm = TopicModeler(n_topics=5, method='nmf')
        dist = tm.fit_transform(dtm, vec)
        assert dist.shape == (len(SAMPLE_DOCS), 5)
        words = tm.get_topic_words(0, n_words=5)
        assert len(words) == 5

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            tm = TopicModeler(method='bert')
            vec = CountVectorizer()
            dtm = vec.fit_transform(SAMPLE_DOCS)
            tm.fit_transform(dtm, vec)

    def test_compare_methods_returns_both(self):
        vec = CountVectorizer(max_features=100, min_df=1)
        dtm = vec.fit_transform(SAMPLE_DOCS)
        tm = TopicModeler(n_topics=5)
        results = tm.compare_methods(dtm, vec)
        assert 'lda' in results
        assert 'nmf' in results
        assert len(results['lda']) == 5
