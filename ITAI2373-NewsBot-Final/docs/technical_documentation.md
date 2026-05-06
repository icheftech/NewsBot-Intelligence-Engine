# NewsBot Intelligence Engine 2.0 — Technical Documentation

> Full technical reference. See `FP_TechnicalDoc_LeroyBrown_LeroyBrown_ITAI2373.pdf` in `/reports/` for the formatted PDF version.

## Architecture

The system is organized into four NLP layers on top of the M1–M8 midterm foundation:

| Layer | Module | Key Technique | Output |
|-------|--------|---------------|--------|
| Foundation | M1–M8 | spaCy, NLTK, TF-IDF, LogReg | 97.6% accuracy, 25,637 NER entities |
| M9 — Advanced Analysis | `src/analysis/` | LDA + NMF (10 topics), VADER | Topic distribution, sentiment profile |
| M10 — Generation | `src/language_models/` | DistilBART, all-MiniLM-L6-v2 | ~60% compression, cosine search |
| M11 — Multilingual | `src/multilingual/` | langdetect, deep-translator | 5+ language auto-classify pipeline |
| M12 — Conversational | `src/conversation/` | 7-intent rule parser, Gradio | Live chat interface |

## Package Structure

```
src/
├── data_processing/
│   ├── text_preprocessor.py   # tokenize, lemmatize, stopword removal
│   ├── feature_extractor.py   # TF-IDF and count vectorizers
│   └── data_validator.py      # DataFrame quality checks
├── analysis/
│   ├── classifier.py          # NewsClassifier (LogReg wrapper)
│   ├── sentiment_analyzer.py  # VADER multi-dimension scoring
│   ├── ner_extractor.py       # spaCy NER (PERSON, ORG, GPE, DATE, MONEY)
│   └── topic_modeler.py       # TopicModeler — LDA and NMF
├── language_models/
│   ├── summarizer.py          # DistilBART abstractive summarization
│   ├── embeddings.py          # SemanticSearchIndex (all-MiniLM-L6-v2)
│   └── generator.py           # Insight generation from article collections
├── multilingual/
│   ├── language_detector.py   # langdetect with seed=42 for reproducibility
│   ├── translator.py          # deep-translator (Google backend)
│   └── cross_lingual_analyzer.py  # detect → translate → classify pipeline
├── conversation/
│   ├── intent_classifier.py   # Rule-based keyword intent detection
│   ├── response_generator.py  # Formatted markdown response builders
│   └── query_processor.py     # QueryProcessor — routes input to backend
└── utils/
    ├── visualization.py       # matplotlib/seaborn plot functions
    ├── evaluation.py          # CV, confusion matrix, classifier comparison
    └── export.py              # CSV/JSON export, topic and stats reports
```

## Classification Performance

| Category | Precision | Recall | F1 | Support |
|----------|-----------|--------|----|---------|
| Business | 0.971 | 0.968 | 0.970 | 80 |
| Entertainment | 0.983 | 0.983 | 0.983 | 60 |
| Politics | 0.957 | 0.957 | 0.957 | 70 |
| Sport | 0.991 | 0.991 | 0.991 | 110 |
| Tech | 0.978 | 0.978 | 0.978 | 80 |
| **Accuracy** | | | **0.976** | 400 |

## Topic Modeling

Both LDA and NMF are implemented in `TopicModeler` with a unified interface:

```python
from src.analysis.topic_modeler import TopicModeler

tm_lda = TopicModeler(n_topics=10, method='lda')
tm_nmf = TopicModeler(n_topics=10, method='nmf')

dist_lda = tm_lda.fit_transform(dtm, vectorizer)
dist_nmf = tm_nmf.fit_transform(dtm, vectorizer)

comparison = tm_lda.compare_methods(dtm, vectorizer)
```

## Intent Parser

Seven intent categories with keyword matching and content extraction:

| Intent | Trigger Keywords | Pipeline |
|--------|-----------------|----------|
| classify | classify, predict, what category | preprocess → TF-IDF → LogReg → VADER |
| summarize | summarize, tldr, summary | token-truncate → DistilBART |
| search | search, find, related to | encode → cosine similarity |
| translate | translate, foreign, non-english | detect → translate → classify |
| stats | stat, accuracy, count, how many | stats_cache lookup |
| topics | topic, keyword, theme | stats_cache lookup |
| help | help, commands, guide | static response |

## Configuration

All tuneable parameters are centralized in `config/settings.py`. Key values:

- `N_TOPICS = 10` — LDA/NMF topic count
- `SEMANTIC_INDEX_SIZE = 500` — articles in embedding index
- `SUMMARIZER_INPUT_TOKEN_LIMIT = 512` — DistilBART input cap
- `LANGDETECT_SEED = 42` — reproducible language detection
