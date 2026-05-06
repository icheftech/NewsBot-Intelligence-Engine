# NewsBot Intelligence Engine 2.0

**ITAI 2373 — Natural Language Processing | Final Project**  
Leroy Brown | Founder & CTO, Southern Shade Technologies | Houston Community College

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/icheftech/NewsBot-Intelligence-Engine/blob/main/ITAI2373-NewsBot-Final/NewsBot_Intelligence_Engine_2.0.ipynb)

---

## Overview

NewsBot 2.0 extends the midterm 97.6%-accuracy classification pipeline with four advanced NLP modules:

| Module | Capability | Key Technology |
|--------|-----------|---------------|
| M9 — Advanced Analysis | LDA + NMF topic modeling, enhanced sentiment | sklearn, pyLDAvis, VADER |
| M10 — Language Generation | Abstractive summarization, semantic search | DistilBART, all-MiniLM-L6-v2 |
| M11 — Multilingual | Language detection + translation + cross-lingual classify | langdetect, deep-translator |
| M12 — Conversational | 7-intent NL query engine with Gradio chat UI | Gradio Blocks |
| Bonus — Web App | Production Next.js dashboard | Next.js 14, Tailwind CSS, Vercel |

---

## Quick Start

```bash
git clone https://github.com/icheftech/NewsBot-Intelligence-Engine.git
cd NewsBot-Intelligence-Engine/ITAI2373-NewsBot-Final
pip install -r requirements.txt
python -m spacy download en_core_web_sm
jupyter notebook NewsBot_Intelligence_Engine_2.0.ipynb
```

Or open directly in Colab using the badge above (recommended).

---

## Repository Structure

```
ITAI2373-NewsBot-Final/
├── NewsBot_Intelligence_Engine_2.0.ipynb  ← Main notebook (M9-M12)
├── requirements.txt
├── config/
│   ├── settings.py                         ← All tuneable parameters
│   └── api_keys_template.txt
├── src/
│   ├── data_processing/
│   │   ├── text_preprocessor.py
│   │   ├── feature_extractor.py
│   │   └── data_validator.py
│   ├── analysis/
│   │   ├── classifier.py                   ← NewsClassifier (97.6% accuracy)
│   │   ├── sentiment_analyzer.py           ← Multi-dim VADER
│   │   ├── ner_extractor.py                ← spaCy NER
│   │   └── topic_modeler.py                ← LDA + NMF
│   ├── language_models/
│   │   ├── summarizer.py                   ← DistilBART (~60% compression)
│   │   ├── embeddings.py                   ← SemanticSearchIndex
│   │   └── generator.py
│   ├── multilingual/
│   │   ├── language_detector.py
│   │   ├── translator.py
│   │   └── cross_lingual_analyzer.py
│   ├── conversation/
│   │   ├── intent_classifier.py
│   │   ├── response_generator.py
│   │   └── query_processor.py
│   └── utils/
│       ├── visualization.py
│       ├── evaluation.py
│       └── export.py
├── notebooks/
│   ├── 01_Data_Exploration.ipynb
│   ├── 02_Advanced_Classification.ipynb
│   ├── 03_Topic_Modeling.ipynb
│   ├── 04_Language_Models.ipynb
│   ├── 05_Multilingual_Analysis.ipynb
│   ├── 06_Conversational_Interface.ipynb
│   └── 07_System_Integration.ipynb
├── tests/
│   ├── test_preprocessing.py
│   ├── test_classification.py
│   ├── test_topic_modeling.py
│   └── test_integration.py
├── data/
│   ├── raw/        ← Original BBC dataset (downloaded at runtime)
│   ├── processed/  ← Cleaned DataFrames
│   ├── models/     ← Saved model artifacts
│   └── results/    ← Analysis outputs and charts
├── docs/
│   ├── technical_documentation.md
│   ├── user_guide.md
│   ├── api_reference.md
│   └── deployment_guide.md
└── reports/
    └── (PDF deliverables)
```

---

## Performance

| Metric | Value |
|--------|-------|
| Classification accuracy | **97.6%** |
| Best F1 score | Sport — 0.991 |
| Dataset | 2,000 BBC articles (5 categories) |
| Summarization compression | ~60% (DistilBART) |
| Semantic search latency | <200ms per query |
| Languages supported | 5+ (ES, FR, DE, PT, ZH) |

---

## Running Tests

```bash
pytest tests/ -v
```

All tests run without GPU or internet access in ~30 seconds.

---

## Individual Contributions

This is an individual project submission.

**Leroy Brown** — All modules (M9–M12), web application, all documentation

See [`docs/technical_documentation.md`](docs/technical_documentation.md) for full system architecture.

---

## License

MIT License — Southern Shade Technologies, 2026
