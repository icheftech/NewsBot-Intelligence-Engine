<div align="center">

<!-- HERO BANNER -->
<img src="https://img.shields.io/badge/NewsBot-Intelligence%20Engine%202.0-C9A84C?style=for-the-badge&logo=rss&logoColor=white" alt="NewsBot 2.0" width="500"/>

# NewsBot Intelligence Engine 2.0

**Production-grade NLP platform that transforms raw news into structured intelligence**

*Classification · Summarization · Semantic Search · Multilingual · Conversational AI · Live RSS · Entity Graphs*

---

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Accuracy](https://img.shields.io/badge/Accuracy-97.6%25-2CA02C?style=flat-square&logo=target&logoColor=white)](#performance)
[![Articles](https://img.shields.io/badge/BBC%20Articles-2%2C000+-C9A84C?style=flat-square&logo=rss&logoColor=white)](#dataset)
[![Languages](https://img.shields.io/badge/Languages-5%2B-9467BD?style=flat-square&logo=google-translate&logoColor=white)](#multilingual)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?style=flat-square&logo=fastapi&logoColor=white)](src/api/main.py)
[![Vercel](https://img.shields.io/badge/Dashboard-Live%20on%20Vercel-000000?style=flat-square&logo=vercel&logoColor=white)](https://newsbot-dashboard.vercel.app)
[![Open In Colab](https://img.shields.io/badge/Open%20in-Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1B-Z3y_La54CYxk7L9Qdf8oGSS1cETzlJ)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

<br/>

[**Live Dashboard →**](https://newsbot-dashboard.vercel.app) · [**API Docs →**](ITAI2373-NewsBot-Final/docs/api_reference.md) · [**Open in Colab →**](https://colab.research.google.com/drive/1B-Z3y_La54CYxk7L9Qdf8oGSS1cETzlJ)

</div>

---

## What is NewsBot 2.0?

NewsBot Intelligence Engine 2.0 is the final-project evolution of a proven NLP classification system — extending a **97.6%-accurate BBC news classifier** into a full intelligence platform with eight integrated capabilities, a REST API, live RSS ingestion, and a production web dashboard.

Built as ITAI 2373 coursework at Houston Community College and simultaneously developed as an **SST enterprise product prototype**, NewsBot 2.0 demonstrates what happens when academic rigor meets production engineering discipline.

> *"I started with a notebook. I ended with a system. And I'm just getting started."*
> — Leroy Brown, Founder & CTO, Southern Shade Technologies

---

## Key Results

| Metric | Value |
|--------|-------|
| Classification accuracy | **97.6%** (Logistic Regression, TF-IDF bigrams) |
| Dataset | 2,000 BBC articles · 5 categories |
| Abstractive compression | **~60%** (DistilBART, CNN/DailyMail fine-tuned) |
| Semantic search index | 500 articles · all-MiniLM-L6-v2 · ~50ms latency |
| Languages supported | **5+** (ES, FR, DE, PT, ZH — auto-detect + translate) |
| NL intents | **7** (classify, summarize, search, translate, stats, topics, help) |
| API endpoints | **11** (FastAPI REST, full Swagger UI) |
| RSS sources | BBC World, Tech, Business, Sport · Reuters |
| Deployment | Vercel (dashboard) · Google Colab (notebooks) |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    INPUT LAYER                                       │
│  BBC Dataset (2,000 articles) · Live RSS Feeds · User Text Input    │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────────┐
│                 FOUNDATION — M1–M8                                   │
│  Text Preprocessing → TF-IDF → POS/Syntax → VADER → LR (97.6%)    │
│  NER (spaCy) · 25,637 entities extracted                            │
└──────┬──────────┬──────────┬──────────┬──────────────────────────── ┘
       │          │          │          │
┌──────▼──┐ ┌────▼────┐ ┌───▼─────┐ ┌─▼────────────┐
│  M9     │ │  M10    │ │  M11    │ │  M12          │
│Advanced │ │Language │ │Multilng │ │Conversational │
│Analysis │ │Gen      │ │Intel    │ │Interface      │
│─────────│ │─────────│ │─────────│ │───────────────│
│LDA/NMF  │ │DistilBART│ │langdetect│ │7-intent NL   │
│10 topics│ │Summary  │ │deep-trans│ │Gradio Blocks │
│pyLDAvis │ │MiniLM   │ │5+ langs │ │share=True URL│
│Bias Det.│ │Sem Search│ │Auto-class│ │Stats sidebar │
└──────┬──┘ └────┬────┘ └───┬─────┘ └─┬────────────┘
       └─────────┴──────────┴──────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────────┐
│                  PRODUCTION LAYER                                    │
│                                                                      │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────────┐ │
│  │  FastAPI REST   │  │  Next.js Dashboard│  │  Research Exts    │ │
│  │  11 endpoints   │  │  Vercel deployed  │  │  NMF · Bias Det   │ │
│  │  /docs Swagger  │  │  6 pages          │  │  Trend Forecast   │ │
│  │  RSS ingestion  │  │  Recharts viz     │  │  Personalization  │ │
│  │  Entity graphs  │  │  SST brand        │  │  Entity Graph     │ │
│  └─────────────────┘  └──────────────────┘  └────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Capabilities

### Core NLP Pipeline (M1–M8)
- **Text Classification** — Logistic Regression on TF-IDF bigrams, 97.6% accuracy across 5 BBC categories
- **POS Tagging** — Grammatical style profiling; Sport leads PROPN density, Tech leads NOUN density
- **Syntax Parsing** — Dependency features, sentence complexity analysis
- **VADER Sentiment** — Multi-dimension scoring (compound, pos, neg, neu) per article
- **Named Entity Recognition** — spaCy extraction of PERSON, ORG, GPE, DATE, MONEY (25,637 entities)

### Advanced Modules (M9–M12)
- **LDA + NMF Topic Modeling** — 10 latent topics, pyLDAvis interactive viz, coherence comparison
- **Abstractive Summarization** — DistilBART (sshleifer/distilbart-cnn-12-6), ~60% compression
- **Semantic Search** — all-MiniLM-L6-v2, 384-dim embeddings, cosine similarity, ~50ms/query
- **Multilingual Pipeline** — langdetect → deep-translator → classify → sentiment, 5+ languages
- **Conversational Interface** — 7-intent NL parser, Gradio Blocks, public share URL

### Production Extensions
- **FastAPI REST Backend** — 11 endpoints, auto Swagger UI at `/docs`, CORS-enabled
- **Live RSS Ingestion** — BBC + Reuters feeds, auto-classify + sentiment on every article at fetch
- **Entity Relationship Graph** — spaCy NER → NetworkX co-occurrence → D3-consumable JSON
- **Research Extensions** — NMF, bias detection (t-test validated), trend forecasting, personalization engine

---

## Quick Start

### Option 1: Google Colab (Zero Setup)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1B-Z3y_La54CYxk7L9Qdf8oGSS1cETzlJ)

```python
# Runtime → Run all
# Dataset downloads automatically via kagglehub — no API keys needed
# Gradio chat interface launches at the final cell with a public URL
```

### Option 2: Local / FastAPI Server

```bash
git clone https://github.com/icheftech/NewsBot-Intelligence-Engine.git
cd NewsBot-Intelligence-Engine/ITAI2373-NewsBot-Final

pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Launch REST API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Swagger UI → http://localhost:8000/docs
```

### Option 3: Live Dashboard

**[newsbot-dashboard.vercel.app](https://newsbot-dashboard.vercel.app)** — No setup required. Classify, summarize, search, translate, and view system metrics in-browser.

---

## Demo Workflow

```python
import requests
BASE = "http://localhost:8000"

# 1 — Classify an article
r = requests.post(f"{BASE}/classify", json={
    "text": "The Prime Minister announced sweeping economic reforms targeting inflation."
})
print(r.json())
# → {"category": "politics", "confidence": 0.9234, "sentiment": {"compound": 0.082, ...}}

# 2 — Summarize
r = requests.post(f"{BASE}/summarize", json={"text": "Arsenal secured the title..."})
print(r.json())
# → {"summary": "Arsenal won the Premier League...", "compression_pct": 61.2}

# 3 — Semantic search
r = requests.post(f"{BASE}/search", json={"query": "AI chip shortage", "top_k": 3})
print(r.json()["results"][0])
# → {"score": 0.8241, "category": "tech", "snippet": "..."}

# 4 — Live RSS — classify today's BBC tech news
r = requests.get(f"{BASE}/rss?feed=bbc_tech&limit=5&analyze=true")
for art in r.json()["articles"]:
    print(f"[{art['category'].upper()}] {art['title']}")

# 5 — Entity graph
r = requests.post(f"{BASE}/entity-graph", json={
    "text": "Elon Musk and Tesla announced a deal with Microsoft..."
})
print(f"{r.json()['node_count']} entities, {r.json()['edge_count']} co-occurrence edges")
```

---

## Repository Structure

```
NewsBot-Intelligence-Engine/
│
├── ITAI2373-NewsBot-Midterm/          ← M1–M8 foundation (97.6% accuracy)
│   └── NewsBot_Intelligence_Engine.ipynb
│
└── ITAI2373-NewsBot-Final/            ← Full 2.0 system
    ├── README.md
    ├── requirements.txt
    │
    ├── notebooks/                     ← 8 tutorial notebooks (all runnable)
    │   ├── 01_Data_Exploration.ipynb
    │   ├── 02_Advanced_Classification.ipynb
    │   ├── 03_Topic_Modeling.ipynb
    │   ├── 04_Language_Models.ipynb
    │   ├── 05_Multilingual_Analysis.ipynb
    │   ├── 06_Conversational_Interface.ipynb
    │   ├── 07_System_Integration.ipynb
    │   └── 08_Research_Extensions.ipynb  ← NMF · Bias · Trend · Personalization
    │
    ├── src/                           ← Modular Python package
    │   ├── api/
    │   │   └── main.py               ← FastAPI backend (11 endpoints)
    │   ├── analysis/
    │   │   ├── classifier.py
    │   │   ├── sentiment_analyzer.py
    │   │   ├── ner_extractor.py
    │   │   └── topic_modeler.py      ← LDA + NMF
    │   ├── language_models/
    │   │   ├── summarizer.py         ← DistilBART
    │   │   └── embeddings.py         ← SentenceTransformers
    │   ├── multilingual/
    │   │   ├── translator.py
    │   │   └── language_detector.py
    │   ├── conversation/
    │   │   ├── query_processor.py
    │   │   └── intent_classifier.py
    │   └── utils/
    │       ├── visualization.py
    │       └── evaluation.py
    │
    ├── newsbot-dashboard/             ← Next.js web app (deployed to Vercel)
    │   ├── app/
    │   ├── components/
    │   └── package.json
    │
    ├── tests/                         ← Unit test suite
    │   ├── test_preprocessing.py
    │   ├── test_classification.py
    │   ├── test_topic_modeling.py
    │   └── test_integration.py
    │
    ├── config/
    │   ├── settings.py
    │   └── api_keys_template.txt
    │
    └── docs/
        ├── technical_documentation.md
        ├── user_guide.md
        ├── api_reference.md
        └── deployment_guide.md
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check + loaded models |
| `GET` | `/stats` | System metrics + category analytics |
| `POST` | `/classify` | Category + confidence + sentiment |
| `POST` | `/summarize` | DistilBART abstractive summary |
| `POST` | `/search` | Semantic similarity search (MiniLM) |
| `POST` | `/sentiment` | Multi-dimension VADER + sentence breakdown |
| `POST` | `/translate` | Language detect + translate + classify |
| `GET` | `/topics` | LDA/NMF topics + category keywords |
| `GET` | `/rss` | Live RSS ingestion + auto-analysis |
| `POST` | `/entity-graph` | NER co-occurrence graph (D3-ready JSON) |
| `POST` | `/analyze` | Full pipeline in one call |

**Full reference →** [`docs/api_reference.md`](ITAI2373-NewsBot-Final/docs/api_reference.md) · **Interactive →** `localhost:8000/docs`

---

## Performance

| Category | Precision | Recall | F1 | Support |
|----------|-----------|--------|----|---------|
| Sport | 0.991 | 0.991 | **0.991** | 110 |
| Entertainment | 0.983 | 0.983 | **0.983** | 60 |
| Tech | 0.978 | 0.978 | **0.978** | 80 |
| Business | 0.971 | 0.968 | **0.970** | 80 |
| Politics | 0.957 | 0.957 | **0.957** | 70 |
| **Overall** | **0.976** | **0.975** | **0.976** | **400** |

*5-fold stratified split · random_state=42 · Logistic Regression (max_iter=1000) on TF-IDF (10K features, bigrams)*

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Core NLP | `nltk` · `spaCy en_core_web_sm` · `scikit-learn` |
| Classification | `LogisticRegression` · `TfidfVectorizer` (bigrams, 10K) |
| Topic Modeling | `sklearn LDA` · `sklearn NMF` · `pyLDAvis` |
| Summarization | `sshleifer/distilbart-cnn-12-6` (HuggingFace) |
| Semantic Search | `all-MiniLM-L6-v2` (SentenceTransformers, 384-dim) |
| Multilingual | `langdetect` · `deep-translator` (Google backend) |
| Conversational UI | `Gradio 4.0+` Blocks API |
| REST API | `FastAPI` · `uvicorn` · `Pydantic v2` |
| Live RSS | `feedparser` · BBC + Reuters |
| Entity Graph | `spaCy` NER → `NetworkX` → D3-ready JSON |
| Web Dashboard | `Next.js 14` · `Tailwind CSS` · `Recharts` |
| Deployment | `Vercel` (dashboard) · Google Colab (notebooks) |
| Data | BBC News Archive via `kagglehub` |

---

## Research Extensions (Bonus)

| Extension | Criterion | Implementation |
|-----------|-----------|----------------|
| NMF Topic Modeling | Cutting-Edge NLP | TF-IDF input, 10 topics, coherence vs LDA |
| Media Bias Detection | Bias Detection | T-test validated VADER skew, word heatmaps |
| Trend Forecasting | Trend Prediction | Linear regression on monthly topic volume, 3-month projection |
| Personalization Engine | Personalization | Preference-weighted semantic search + query history blending |
| Real-Time Processing | Real-Time Processing | FastAPI `/rss` — live BBC/Reuters auto-classify |
| Entity Relationship Graph | Novel Application | spaCy → NetworkX → degree centrality ranking |

---

## Roadmap

- [ ] DistilBERT fine-tuned classifier (target: Politics F1 > 0.97)
- [ ] BERTopic replacing LDA for neural topic clustering
- [ ] FAISS index scaling to 1M+ articles
- [ ] Real-time RSS pipeline with timestamp drift detection
- [ ] FastAPI authentication + rate limiting for production
- [ ] SST enterprise deployment for federal OSINT analyst workflows

---

## Project Context

This project was developed for **ITAI 2373 — Natural Language Processing** at Houston Community College as the Final Project, and simultaneously serves as an R&D prototype for [Southern Shade Technologies](https://southernshadetechnologies.com) — an AI infrastructure and consulting firm.

The same architectural patterns demonstrated here — agentic pipelines, semantic retrieval, multilingual intelligence — are directly applicable to SST's federal contracting pipeline, including CFIC/ARCYBER intelligence automation work.

---

## Author

**Leroy Brown**
Founder & CTO — Southern Shade Technologies
AI & Robotics · Houston Community College

[![GitHub](https://img.shields.io/badge/GitHub-icheftech-181717?style=flat-square&logo=github)](https://github.com/icheftech)
[![SST](https://img.shields.io/badge/SST-southernshadetechnologies.com-C9A84C?style=flat-square)](https://southernshadetechnologies.com)

---

<div align="center">

*ITAI 2373 — Natural Language Processing | Houston Community College | May 2026*

**NewsBot Intelligence Engine 2.0 — Intelligent Systems Built to Scale**

</div>
