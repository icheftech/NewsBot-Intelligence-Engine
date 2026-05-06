"""
Unit tests for text preprocessing and data validation.
"""

import pytest
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_processing.text_preprocessor import preprocess, preprocess_corpus
from src.data_processing.data_validator import validate_dataframe, clean_dataframe


class TestPreprocess:
    def test_returns_string_by_default(self):
        result = preprocess("The Prime Minister announced new policies.")
        assert isinstance(result, str)

    def test_lowercases_text(self):
        result = preprocess("HELLO WORLD")
        assert result == result.lower()

    def test_removes_punctuation(self):
        result = preprocess("Hello, world! How are you?")
        assert ',' not in result and '!' not in result

    def test_removes_stopwords(self):
        result = preprocess("the quick brown fox")
        assert 'the' not in result.split()

    def test_returns_tokens_when_flag_set(self):
        result = preprocess("The government announced new measures", return_tokens=True)
        assert isinstance(result, list)

    def test_short_text_handled(self):
        result = preprocess("AI")
        assert isinstance(result, str)

    def test_empty_string(self):
        result = preprocess("")
        assert isinstance(result, str)


class TestPreprocessCorpus:
    def test_returns_list(self):
        texts = ["Hello world", "Tech news is great"]
        result = preprocess_corpus(texts)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_token_mode(self):
        result = preprocess_corpus(["Hello world"], return_tokens=True)
        assert isinstance(result[0], list)


class TestDataValidator:
    def _make_df(self, n=10):
        return pd.DataFrame({
            'category': ['business', 'sport', 'tech', 'politics', 'entertainment'] * (n // 5),
            'text': ['This is a longer test article text with sufficient length for validation purposes.'] * n,
        })

    def test_valid_dataframe_passes(self):
        df = self._make_df()
        result = validate_dataframe(df)
        assert result['passed'] is True
        assert len(result['issues']) == 0

    def test_missing_column_detected(self):
        df = pd.DataFrame({'category': ['sport']})
        result = validate_dataframe(df)
        assert not result['passed']
        assert any('text' in issue for issue in result['issues'])

    def test_null_values_detected(self):
        df = self._make_df()
        df.loc[0, 'text'] = None
        result = validate_dataframe(df)
        assert not result['passed']

    def test_unknown_category_detected(self):
        df = self._make_df()
        df.loc[0, 'category'] = 'weather'
        result = validate_dataframe(df)
        assert not result['passed']

    def test_clean_dataframe_removes_nulls(self):
        df = self._make_df()
        df.loc[0, 'text'] = None
        cleaned = clean_dataframe(df)
        assert cleaned.isnull().sum().sum() == 0

    def test_clean_dataframe_filters_categories(self):
        df = self._make_df()
        df.loc[0, 'category'] = 'weather'
        cleaned = clean_dataframe(df)
        assert 'weather' not in cleaned['category'].values
