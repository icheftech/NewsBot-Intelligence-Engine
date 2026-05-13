# Individual Contributions — NewsBot Intelligence Engine 2.0

**Submitter:** Leroy Brown  
**Group:** Solo  
**Course:** ITAI 2373 — Introduction to Artificial Intelligence  
**Institution:** Houston Community College  
**Date:** May 2026

---

## Submission Type

This is a **solo submission**. All work below was completed independently by Leroy Brown.

---

## Contributions Summary

### Data Engineering
- Designed and implemented `src/data_processing/data_loader.py` — kagglehub download with automatic CSV discovery and column normalization
- Implemented `src/data_processing/data_validator.py` — schema validation, null checks, and category filtering
- Implemented `src/data_processing/text_preprocessor.py` — full preprocessing pipeline (lowercase, URL removal, tokenization, stopword removal, Porter stemming)
- Implemented `src/data_processing/preprocessor.py` — public API wrapper for the preprocessing pipeline
- Implemented `src/data_processing/feature_extractor.py` — TF-IDF, bigram, and embedding feature extraction

### Analysis Modules
- Implemented `src/analysis/classifier.py` — `NewsClassifier` class with Logistic Regression, train/test/evaluate pipeline, 5-fold CV, and 3-model comparison (LogReg, Naive Bayes, LinearSVC)
- Implemented `src/analysis/topic_modeler.py` — `TopicModeler` class supporting LDA and NMF with pyLDAvis visualization and `compare_methods()`
- Implemented `src/analysis/sentiment_analyzer.py` — VADER-based `score_text()` with compound/pos/neg/neu breakdown
- Implemented `src/analysis/ner_extractor.py` — spaCy NER for PERSON, ORG, GPE, DATE, MONEY entities

### Language Models
- Implemented `src/language_models/summarizer.py` — `summarize_with_stats()` using DistilBART (sshleifer/distilbart-cnn-12-6) with compression metrics
- Implemented `src/language_models/embeddings.py` — `SemanticSearchIndex` class using all-MiniLM-L6-v2 (384-dim) with cosine similarity search
- Implemented `src/language_models/generator.py` — response generation utilities

### Multilingual Pipeline
- Implemented `src/multilingual/language_detector.py` — langdetect with seed=42 for reproducibility
- Implemented `src/multilingual/translator.py` — `translate_and_detect()` using deep-translator GoogleTranslator
- Implemented `src/multilingual/cross_lingual_analyzer.py` — cross-language coverage comparison

### Conversational Interface
- Implemented `src/conversation/intent_classifier.py` — `classify_intent()` with 7 intents (classify, summarize, search, translate, stats, topics, help)
- Implemented `src/conversation/query_processor.py` — `QueryProcessor` class routing all 7 intents with multi-turn history support
- Implemented `src/conversation/response_generator.py` — formatted response generation for all intent types
- Built Gradio ChatInterface frontend with `demo.launch(share=True)`

### Utilities
- Implemented `src/utils/visualization.py` — advanced matplotlib/seaborn plotting functions
- Implemented `src/utils/evaluation.py` — model evaluation metrics and cross-validation utilities
- Implemented `src/utils/export.py` — JSON/CSV/PNG report generation

### Notebooks (7 core + 1 research bonus)
- `01_Data_Exploration.ipynb` — EDA, distribution analysis, word clouds
- `02_Advanced_Classification.ipynb` — 3-model comparison, confusion matrix, per-category F1
- `03_Topic_Modeling.ipynb` — LDA + NMF, pyLDAvis, topic-category mapping
- `04_Language_Models.ipynb` — DistilBART summarization, Sentence-BERT semantic search
- `05_Multilingual_Analysis.ipynb` — 5-language detection and translation pipeline
- `06_Conversational_Interface.ipynb` — QueryProcessor demo, 7-intent test suite, Gradio launch
- `07_System_Integration.ipynb` — 10-stage timed pipeline, performance dashboard, final report
- `08_Research_Extensions.ipynb` — R1: NMF coherence, R2: Bias detection, R3: Trend prediction, R4: Personalization

### Web Application (30 bonus points)
- Designed and implemented `web/api.py` — FastAPI backend with 8 REST endpoints and full NLP pipeline initialization
- Designed and implemented `web/static/index.html` — Single-page application with Tailwind CSS, 5-tab interface, live stats sidebar

### Research Extensions (20 bonus points)
- R1: Hybrid LDA+NMF coherence comparison with statistical scoring
- R2: VADER-based media bias detection with t-test validation and word heatmaps
- R3: NMF time-series trend prediction with linear regression forecasting
- R4: Personalized semantic search with preference weighting and diversity re-ranking

### Documentation & Reports
- `README.md` — comprehensive project overview and setup guide
- `docs/technical_documentation.md` — full architecture and API reference
- `docs/user_guide.md` — end-user instructions
- `docs/api_reference.md` — complete endpoint and function documentation
- `docs/deployment_guide.md` — production deployment instructions
- `FP_TechnicalDoc_LeroyBrown_Solo_ITAI2373.pdf` — 14-section technical documentation
- `FP_ExecutiveSummary_LeroyBrown_Solo_ITAI2373.pdf` — business-focused project overview
- `FP_ReflectiveJournal_Solo_ITAI2373.pdf` — personal reflective journal
- `FP_Presentation_LeroyBrown_Solo_ITAI2373.pptx` — 14-slide professional presentation

### Testing
- `tests/test_preprocessing.py` — preprocessing pipeline unit tests
- `tests/test_classification.py` — classifier fit, predict, evaluate, CV tests
- `tests/test_topic_modeling.py` — LDA and NMF fit, topic words, compare_methods tests
- `tests/test_integration.py` — 7 end-to-end pipeline integration tests

---

## Key Metrics Achieved

| Metric | Value |
|--------|-------|
| Classification accuracy | 97.6% (Logistic Regression) |
| Summarization compression | 82.7% average |
| Languages supported | 5 (ES, FR, DE, PT, ZH) |
| Conversational intents | 7 |
| NLP pipelines integrated | 8 |
| Test coverage | 4 test files, 15+ test cases |
| NMF topic coherence | 0.3828 (vs LDA 0.3518) |
| Bias detection | Politics: 44.9% negative — statistically significant (t-test) |
