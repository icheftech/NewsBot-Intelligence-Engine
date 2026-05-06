"""
NewsBot Intelligence Engine 2.0 — Configuration
"""

# Dataset
DATASET_SIZE = 2000
RANDOM_STATE = 42
CATEGORIES = ['business', 'entertainment', 'politics', 'sport', 'tech']
CATEGORY_PALETTE = {
    'business': '#1f77b4',
    'entertainment': '#e377c2',
    'politics': '#d62728',
    'sport': '#2ca02c',
    'tech': '#ff7f0e',
}

# TF-IDF
TFIDF_MAX_FEATURES = 10000
TFIDF_NGRAM_RANGE = (1, 2)

# Topic Modeling
N_TOPICS = 10
LDA_MAX_ITER = 20
NMF_MAX_ITER = 400
COUNT_VEC_MAX_FEATURES = 5000
COUNT_VEC_MIN_DF = 5
COUNT_VEC_MAX_DF = 0.9

# Summarization
SUMMARIZER_MODEL = 'sshleifer/distilbart-cnn-12-6'
SUMMARY_MAX_LENGTH = 130
SUMMARY_MIN_LENGTH = 30
SUMMARIZER_INPUT_TOKEN_LIMIT = 512

# Semantic Search
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
SEMANTIC_INDEX_SIZE = 500
SEMANTIC_ARTICLE_TOKEN_LIMIT = 256
SEMANTIC_TOP_K = 5

# Sentiment
VADER_POSITIVE_THRESHOLD = 0.05
VADER_NEGATIVE_THRESHOLD = -0.05

# Language Detection
LANGDETECT_SEED = 42
LANG_DETECT_SAMPLE_SIZE = 400
LANG_DETECT_TEXT_LIMIT = 300

# Translation
TRANSLATION_TARGET_LANG = 'en'

# Gradio
GRADIO_SHARE = True
