# NewsBot 2.0 — Performance Summary
**ITAI 2373 | Leroy Brown**

## Classification Performance

| Model | Accuracy | CV Mean |
|-------|----------|---------|
| Logistic Regression (TF-IDF) | **97.6%** | 97.4% |
| Naive Bayes (TF-IDF) | 94.1% | 93.8% |
| LinearSVC (TF-IDF) | 97.1% | 96.9% |

## Per-Category F1 Scores (Logistic Regression)

| Category | Precision | Recall | F1 |
|----------|-----------|--------|----|
| Business | 0.97 | 0.96 | 0.97 |
| Entertainment | 0.98 | 0.99 | 0.98 |
| Politics | 0.97 | 0.97 | 0.97 |
| Sport | 1.00 | 0.99 | 0.99 |
| Tech | 0.97 | 0.98 | 0.97 |

## Summarization Compression

| Category | Original Words | Summary Words | Compression % |
|----------|---------------|---------------|---------------|
| Business | 382 | 58 | 84.8% |
| Entertainment | 337 | 62 | 81.6% |
| Politics | 406 | 71 | 82.5% |
| Sport | 344 | 64 | 81.4% |
| Tech | 358 | 60 | 83.2% |
| **Average** | **365** | **63** | **82.7%** |

## Sentiment Distribution

| Polarity | Compound Range | % of Corpus |
|----------|---------------|-------------|
| Positive | ≥ 0.05 | 38.2% |
| Negative | ≤ -0.05 | 28.7% |
| Neutral | (-0.05, 0.05) | 33.1% |

## Intent Classification Accuracy

| Intent | Trigger Pattern | Routing |
|--------|----------------|---------|
| classify | "classify:" prefix | TF-IDF + LogReg |
| summarize | "summarize:" prefix | DistilBART |
| search | "search:" prefix | Sentence-BERT |
| translate | "translate:" prefix | langdetect + deep-translator |
| stats | "stat/count/accuracy" | Cached analytics |
| topics | "topic/keyword/theme" | LDA/NMF words |
| help | "help/commands" | Static response |

All 7 intents: **100% routing accuracy** on test suite.

## Model Configuration

| Component | Model / Config |
|-----------|---------------|
| Vectorizer | TF-IDF, 10K features, (1,2)-grams, min_df=2 |
| Classifier | LogisticRegression, max_iter=1000, seed=42 |
| Summarizer | sshleifer/distilbart-cnn-12-6 |
| Embeddings | all-MiniLM-L6-v2 (384 dims) |
| Topic model | LDA + NMF, 10 topics each, seed=42 |
| Sentiment | VADER (compound + pos/neg/neu) |
| NER | spaCy en_core_web_sm |
| Translator | deep-translator (Google, auto-detect) |
