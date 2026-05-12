# NewsBot Intelligence Engine 2.0
## Final Project Report — ITAI 2373: Introduction to Artificial Intelligence

**Student:** Leroy Brown  
**Institution:** Houston Community College  
**Course:** ITAI 2373 — Introduction to Artificial Intelligence  
**Submission:** Final Project  
**Date:** May 2026  

---

## Executive Summary

NewsBot Intelligence Engine 2.0 is a full-stack AI-powered news analysis system built on the BBC News Archive dataset (2,225 articles across 5 categories). The system integrates eight NLP pipelines — classification, topic modeling, abstractive summarization, semantic search, sentiment analysis, named entity recognition, multilingual translation, and a conversational interface — into a single modular Python codebase deployable via Google Colab and Gradio.

**Key Results:**

| Metric | Value |
|--------|-------|
| Dataset | 2,225 BBC articles, 5 categories |
| Classification accuracy | 97.6% (Logistic Regression + TF-IDF) |
| Topic models | LDA + NMF (10 topics each) |
| Summarization compression | ~60% average |
| Search embedding dimensions | 384 (Sentence-BERT) |
| Languages supported | ES, FR, DE, PT, ZH + auto-detect |
| Conversational intents | 7 (classify, summarize, search, translate, stats, topics, help) |

---

## 1. Problem Statement

News consumers face information overload — thousands of articles published daily with no intelligent layer to categorize, summarize, or contextualize them. This project builds a practical AI system that:

1. Automatically categorizes news articles into 5 domains
2. Extracts latent topics from large corpora
3. Generates abstractive summaries at ~60% compression
4. Enables semantic (meaning-based) article search
5. Performs multi-dimension sentiment analysis
6. Extracts named entities (persons, organizations, locations)
7. Translates and classifies foreign-language content
8. Provides a conversational natural language interface

---

## 2. Dataset

**Source:** BBC News Archive — `hgultekin/bbcnewsarchive` (Kaggle)  
**Size:** 2,225 articles | **Categories:** 5 | **Format:** CSV (`category`, `text`)

| Category | Articles | Avg Words | Avg Sentiment |
|----------|----------|-----------|---------------|
| Business | 510 | 382 | +0.041 |
| Entertainment | 386 | 337 | +0.089 |
| Politics | 417 | 406 | -0.023 |
| Sport | 511 | 344 | +0.112 |
| Tech | 401 | 358 | +0.067 |

**Preprocessing pipeline:**
- Lowercase normalization
- URL / punctuation removal
- NLTK tokenization
- Stopword removal (English)
- Porter stemming

---

## 3. Methodology

### 3.1 Text Classification

**Algorithm:** Logistic Regression (`sklearn`, `max_iter=1000`, `random_state=42`)  
**Features:** TF-IDF (`max_features=10,000`, `ngram_range=(1,2)`, `min_df=2`)  
**Evaluation:** 80/20 stratified train/test split + 5-fold cross-validation

**Results:**

| Category | Precision | Recall | F1 |
|----------|-----------|--------|----|
| Business | 0.97 | 0.96 | 0.97 |
| Entertainment | 0.98 | 0.99 | 0.98 |
| Politics | 0.97 | 0.97 | 0.97 |
| Sport | 1.00 | 0.99 | 0.99 |
| Tech | 0.97 | 0.98 | 0.97 |
| **Overall** | **0.978** | **0.978** | **0.978** |

Compared algorithms: Logistic Regression outperformed Naive Bayes (94.1%) and LinearSVC (97.1%).

### 3.2 Topic Modeling

Two complementary methods were implemented and compared:

**LDA (Latent Dirichlet Allocation)**
- Probabilistic generative model
- 10 topics, `learning_method='online'`, `max_iter=10`
- Captures overlapping topic distributions per document
- Visualized with pyLDAvis interactive interface

**NMF (Non-negative Matrix Factorization)**
- Deterministic matrix factorization
- 10 topics, `init='nndsvda'`, `max_iter=200`
- Produces sparser, more interpretable topics
- Better separation between hard-news categories

Sample LDA topics extracted:

| Topic | Top Keywords |
|-------|-------------|
| Politics | government, minister, party, election, policy |
| Sport | game, match, win, player, team |
| Business | market, company, growth, economy, profit |
| Tech | technology, software, mobile, internet, digital |
| Entertainment | film, music, award, star, show |

### 3.3 Abstractive Summarization

**Model:** `sshleifer/distilbart-cnn-12-6` (HuggingFace Transformers)  
**Architecture:** DistilBART — distilled BART fine-tuned on CNN/DailyMail  
**Method:** Token-limited abstractive generation (not extractive)

| Category | Avg Original Words | Avg Summary Words | Compression |
|----------|--------------------|-------------------|-------------|
| Business | 382 | 58 | 84.8% |
| Entertainment | 337 | 62 | 81.6% |
| Politics | 406 | 71 | 82.5% |
| Sport | 344 | 64 | 81.4% |
| Tech | 358 | 60 | 83.2% |

### 3.4 Semantic Search

**Model:** `sentence-transformers/all-MiniLM-L6-v2`  
**Dimensions:** 384 | **Index size:** 500 articles  
**Similarity metric:** Cosine similarity  

Articles are encoded into dense 384-dimensional vectors. At query time the query is encoded and compared against all indexed articles. Top-k results returned with similarity scores and snippets.

Advantage over keyword search: retrieves semantically related articles even when exact query terms are absent.

### 3.5 Sentiment Analysis

**Engine:** VADER (Valence Aware Dictionary and sEntiment Reasoner) — `nltk.sentiment.vader`  

VADER returns four scores per article:
- `compound` — normalized overall sentiment (-1.0 to +1.0)
- `pos` — proportion of positive tokens
- `neg` — proportion of negative tokens
- `neu` — proportion of neutral tokens

**Corpus-level findings:**
- 38.2% Positive (compound ≥ 0.05)
- 28.7% Negative (compound ≤ -0.05)
- 33.1% Neutral

Sport articles are most positive (+0.112); Politics most negative (-0.023).

### 3.6 Named Entity Recognition

**Library:** spaCy `en_core_web_sm`  
**Entity types:** PERSON, ORG, GPE (geopolitical), DATE, MONEY

Used to extract key actors, organizations, and locations from articles for metadata enrichment and downstream analysis.

### 3.7 Multilingual Support

**Detection:** `langdetect` (`DetectorFactory.seed=42` for reproducibility)  
**Translation:** `deep-translator` — Google Translate backend (no API key required)  
**Pipeline:** Detect language → translate to English → classify → sentiment score

Languages tested: Spanish (ES), French (FR), German (DE), Portuguese (PT), Chinese (ZH).

Detection accuracy on test corpus: 100% on sampled foreign articles.

### 3.8 Conversational Interface

**Framework:** Gradio `ChatInterface` (Blocks API)  
**Intent routing:** Rule-based keyword classifier (7 intents)

| Intent | Trigger Keywords | Backend |
|--------|-----------------|---------|
| classify | "classify:", "what category", "predict" | TF-IDF + LogReg |
| summarize | "summarize:", "tldr:", "summary:" | DistilBART |
| search | "search:", "find articles", "related to" | Sentence-BERT |
| translate | "translate:", "in spanish/french" | langdetect + deep-translator |
| stats | "stat", "how many", "accuracy", "distribution" | Pre-computed cache |
| topics | "topic", "keyword", "theme" | LDA/NMF word lists |
| help | "help", "what can you", "commands" | Static text |

---

## 4. System Architecture

```
ITAI2373-NewsBot-Final/
├── config/settings.py          # Centralized hyperparameters
├── src/
│   ├── data_processing/        # Loader, preprocessor, validator
│   ├── analysis/               # Classifier, topic modeler, sentiment
│   ├── language_models/        # Summarizer, embeddings (semantic search)
│   ├── multilingual/           # Language detector, translator
│   └── conversation/           # Intent classifier, response generator, query processor
├── notebooks/                  # 7 Jupyter notebooks (01–07)
├── tests/                      # pytest unit + integration tests
├── data/
│   ├── raw/                    # Downloaded BBC dataset
│   └── results/                # Pipeline outputs (JSON, CSV, PNG)
└── reports/                    # This document
```

**Key design decisions:**
- Modular `src/` package — each capability is independently importable
- Centralized `config/settings.py` — all hyperparameters in one place
- `QueryProcessor` orchestrates all backends via single `process()` call
- Notebooks import from `src/` (no code duplication)

---

## 5. Testing

**Framework:** pytest  
**Test files:** `tests/test_classifier.py`, `tests/test_topic_modeling.py`, `tests/test_sentiment.py`, `tests/test_search.py`, `tests/test_integration.py`

| Test Class | Coverage |
|------------|----------|
| `TestNewsClassifier` | fit, predict, evaluate, cross-validation |
| `TestTopicModelerLDA` | fit, get_topic_words, dominant topics |
| `TestTopicModelerNMF` | fit, topics, invalid method raises |
| `TestSentimentAnalyzer` | score_text, pos/neg/neutral classification |
| `TestSemanticSearch` | build index, search returns top-k |
| `TestEndToEndPipeline` | preprocess→TF-IDF, validate→clean, full classify flow |

---

## 6. Tools & Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| scikit-learn | ≥1.3 | TF-IDF, Logistic Regression, LDA, NMF, metrics |
| transformers | ≥4.35 | DistilBART summarization |
| sentence-transformers | ≥2.2 | Sentence-BERT embeddings |
| nltk | ≥3.8 | VADER sentiment, tokenization |
| spacy | ≥3.7 | Named entity recognition |
| langdetect | ≥1.0.9 | Language detection |
| deep-translator | ≥1.11 | Google Translate (no key required) |
| gradio | ≥4.0 | Conversational UI |
| pyLDAvis | ≥3.4 | Interactive LDA visualization |
| kagglehub | ≥0.2 | Automatic dataset download |
| pandas | ≥2.0 | Data manipulation |
| matplotlib / seaborn | ≥3.7 | Visualizations |
| pytest | ≥7.4 | Unit and integration testing |

---

## 7. Results Summary

The NewsBot 2.0 system successfully demonstrates:

1. **High-accuracy classification** — 97.6% on BBC corpus, outperforming course baseline
2. **Dual topic modeling** — LDA and NMF both implemented and compared
3. **Effective summarization** — ~82% average compression while preserving key information
4. **Semantic search** — meaning-based retrieval outperforms keyword matching on ambiguous queries
5. **Robust multilingual pipeline** — auto-detect + translate + classify in one call
6. **Functional chatbot** — all 7 intents correctly routed in conversation tests

---

## 8. References

1. BBC News Archive Dataset — Huseyin Gultekin, Kaggle (2023)
2. Devlin et al. — BERT: Pre-training of Deep Bidirectional Transformers (2019)
3. Reimers & Gurevych — Sentence-BERT: Sentence Embeddings (2019)
4. Lewis et al. — BART: Denoising Sequence-to-Sequence Pre-training (2020)
5. Hutto & Gilbert — VADER: A Parsimony-Based Sentiment Analysis (2014)
6. Blei, Ng & Jordan — Latent Dirichlet Allocation, JMLR (2003)
7. Lee & Seung — Algorithms for Non-negative Matrix Factorization, NIPS (2000)
