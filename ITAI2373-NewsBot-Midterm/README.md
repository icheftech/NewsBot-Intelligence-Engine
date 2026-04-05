# 🤖 NewsBot Intelligence Engine
### ITAI 2373 — Mid-Term Project | Leroy Brown

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![spaCy](https://img.shields.io/badge/spaCy-3.8-09a3d5)](https://spacy.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-f89939)](https://scikit-learn.org)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1B-Z3y_La54CYxk7L9Qdf8oGSS1cETzlJ)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> End-to-end NLP pipeline that automatically classifies news articles, extracts named entities, scores sentiment, and surfaces linguistic patterns — demonstrating all Modules 1–8 of ITAI 2373.

---

## 📋 Project Overview

NewsBot transforms raw BBC news articles into structured intelligence through a complete NLP pipeline:

```
Raw Article Text
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  Module 2: Text Preprocessing  (lowercase → tokenize → lemmatize) │
└──────────────────────┬──────────────────────────────────────┘
                       │
       ┌───────────────┼──────────────────┐
       ▼               ▼                  ▼
Module 3: TF-IDF   Module 4: POS    Module 5: Syntax
Feature Extraction  Tag Analysis     Dependency Parse
       │               │                  │
       └───────────────┴──────────────────┘
                       │
       ┌───────────────┼──────────────────┐
       ▼               ▼                  ▼
Module 6: Sentiment  Module 7: ML    Module 8: NER
(VADER)              Classification  Entity Extraction
       │               │                  │
       └───────────────┴──────────────────┘
                       │
                       ▼
          📊 Structured News Intelligence
             + 🚀 Gradio Live Demo
```

---

## 🏗️ System Components (All 8 Modules)

| Module | Component | Technique |
|--------|-----------|-----------|
| 1 | Business Context | Use-case analysis, target user mapping |
| 2 | Text Preprocessing | Tokenisation, stopword removal, lemmatisation |
| 3 | TF-IDF Analysis | Feature extraction, category term heatmaps |
| 4 | POS Tagging | Grammatical style profiling by category |
| 5 | Syntax Parsing | Dependency features, sentence complexity |
| 6 | Sentiment | VADER compound scoring, violin + stacked charts |
| 7 | Classification | LogReg / Naive Bayes / LinearSVC comparison |
| 8 | NER | PERSON, ORG, GPE, DATE, MONEY extraction |
| Bonus | Gradio Demo | Interactive live inference interface |

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| Dataset | 2,000 BBC News articles (sampled from 2,225) |
| Categories | 5 — Business, Entertainment, Politics, Sport, Tech |
| Best Model | Logistic Regression — **97.6% accuracy** |
| Runner-up | Linear SVC — 97.6% accuracy |
| Total Entities Extracted | 25,637 across 5 entity types |
| Top Sentiment Category | Entertainment (+0.532) |
| Most Negative Category | Politics (+0.072) |
| Sport F1 Score | 0.991 — highest per-class performance |

### Model Comparison

| Model | Accuracy |
|-------|----------|
| Logistic Regression | **97.6%** |
| Linear SVC | 97.6% |
| Multinomial NB | 97.2% |

### Per-Class F1 Scores

| Category | LogReg | NB | LinearSVC |
|----------|--------|----|-----------|
| Business | 0.970 | 0.961 | 0.969 |
| Entertainment | 0.983 | 0.983 | 0.983 |
| Politics | 0.957 | 0.958 | 0.968 |
| Sport | 0.991 | 0.991 | 0.991 |
| Tech | 0.978 | 0.966 | 0.967 |

---

## 🚀 How to Run

### Option 1: Google Colab (Recommended)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1B-Z3y_La54CYxk7L9Qdf8oGSS1cETzlJ)

1. Click the badge above to open in Colab
2. `Runtime → Run all`
3. Dataset downloads automatically via `kagglehub` — no API key needed
4. Gradio demo launches at the end with a public URL

### Option 2: Local Setup
```bash
git clone https://github.com/icheftech/NewBot-Intelligence-Engine.git
cd NewBot-Intelligence-Engine
pip install -r requirements.txt
python -m spacy download en_core_web_sm
jupyter notebook NewsBot_Intelligence_Engine.ipynb
```

---

## 📁 Repository Structure

```
NewBot-Intelligence-Engine/
├── NewsBot_Intelligence_Engine.ipynb   ← Main notebook (all 8 modules)
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 📦 Dependencies

```
nltk>=3.8
spacy>=3.8
scikit-learn>=1.4
pandas>=2.0
numpy>=1.26
matplotlib>=3.8
seaborn>=0.13
gradio>=4.0
kagglehub>=0.2
```

---

## 💡 Key Insights

**TF-IDF** revealed near-zero vocabulary overlap between categories — *labour*, *blair*, *election* are exclusive to Politics; *mobile*, *phone*, *microsoft* to Tech; *film*, *oscar*, *award* to Entertainment.

**Sentiment** showed all 5 categories are net-positive on this BBC corpus (2004–2005 era), with Entertainment highest (+0.532) and Politics lowest (+0.072) — reflecting award coverage vs. contentious election-year reporting.

**POS tagging** confirmed Tech articles are the most NOUN-dense (0.414) while Sport leads PROPN (0.302) due to player and team name density.

**NER** extracted 25,637 entities — Sport tops PERSON density (6.0 avg per article), Business tops MONEY density (1.6 avg per article), Politics dominates ORG mentions (Labour: 159, EU: 55).

**Classification** achieved 97.6% accuracy across 5 categories — Sport was the easiest to classify (F1: 0.991), Politics the hardest (F1: 0.957) due to overlap with Business on economic policy stories.

---

## 🔮 Future Enhancements

- [ ] Fine-tuned BERT/DistilBERT for improved contextual classification
- [ ] LDA/BERTopic for within-category trend discovery
- [ ] Real-time RSS ingestion pipeline with timestamp tracking
- [ ] Entity co-occurrence relationship graph (NetworkX)
- [ ] FastAPI production wrapper for newsroom CMS integration
- [ ] Class-weight balancing to further improve Politics recall

---

## 👤 Author

**Leroy Brown**  
Founder & CTO, Southern Shade Technologies  
AI & Robotics — Houston Community College  
[GitHub](https://github.com/icheftech) | [SST](https://southernshadetechnologies.com)

---

*ITAI 2373 — Natural Language Processing | Houston Community College*
