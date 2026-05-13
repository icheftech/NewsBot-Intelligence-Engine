# Deployment Instructions — NewsBot Intelligence Engine 2.0

**ITAI 2373 Final Project | Leroy Brown | Houston Community College | May 2026**

---

## Option 1: Google Colab (Recommended for Grading)

All notebooks are designed to run end-to-end in Google Colab with no local setup.

### Steps

1. Open [github.com/icheftech/NewsBot-Intelligence-Engine](https://github.com/icheftech/NewsBot-Intelligence-Engine)
2. Navigate to `ITAI2373-NewsBot-Final/notebooks/`
3. Click any notebook → **Open in Colab** button (top of GitHub preview)
4. In Colab: **Runtime → Run All**

The notebooks install all dependencies in Cell 1 and download the BBC dataset automatically via `kagglehub`.

### Recommended Run Order
```
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 (optional bonus)
```

Each notebook is self-contained and can also run independently.

---

## Option 2: Local Python Environment

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/icheftech/NewsBot-Intelligence-Engine.git
cd NewsBot-Intelligence-Engine/ITAI2373-NewsBot-Final

# Install dependencies
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; [nltk.download(p) for p in ['punkt','stopwords','wordnet','vader_lexicon','punkt_tab']]"
```

### Run Notebooks

```bash
jupyter notebook notebooks/
```

---

## Option 3: Web Application (FastAPI)

Serves the full NLP pipeline via REST API + browser frontend.

### Start the Server

```bash
cd ITAI2373-NewsBot-Final
uvicorn web.api:app --reload --port 8000
```

Open **http://localhost:8000** in your browser.

> First launch takes 1–2 minutes while the pipeline initializes (downloads dataset, fits TF-IDF, builds semantic index). The UI shows a loading indicator until ready.

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Pipeline readiness check |
| POST | `/api/chat` | Full conversational interface |
| POST | `/api/classify` | Category + confidence + sentiment |
| POST | `/api/summarize` | Abstractive summary + compression % |
| POST | `/api/search` | Semantic similarity search |
| POST | `/api/translate` | Language detect + translate + classify |
| GET | `/api/stats` | Dataset analytics |
| GET | `/api/topics` | LDA/NMF topic keywords |

### Run in Google Colab with Public URL

```python
!pip install fastapi uvicorn python-multipart nest-asyncio pyngrok

import nest_asyncio
from pyngrok import ngrok
nest_asyncio.apply()

ngrok.set_auth_token("YOUR_NGROK_TOKEN")  # Free at ngrok.com
public_url = ngrok.connect(8000)
print(f"Public URL: {public_url}")

import uvicorn
uvicorn.run("web.api:app", host="0.0.0.0", port=8000)
```

---

## Environment Variables / Configuration

All configuration is managed in `config/settings.py`:

```python
DATASET_SIZE = 2000          # Articles sampled from BBC corpus
RANDOM_STATE = 42            # Reproducibility seed
N_TOPICS = 10                # LDA/NMF topic count
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
SUMMARIZER_MODEL = 'sshleifer/distilbart-cnn-12-6'
LANGDETECT_SEED = 42
SEMANTIC_INDEX_SIZE = 500    # Articles in semantic search index
```

No API keys required. All models are downloaded from Hugging Face Hub on first run.

See `config/api_keys_template.txt` for optional HuggingFace token setup (improves download speed).

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 4 GB | 8 GB |
| Storage | 3 GB | 5 GB |
| GPU | Not required | Optional (speeds up DistilBART) |
| Python | 3.10 | 3.10–3.12 |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'src'`**
```bash
# Run from the ITAI2373-NewsBot-Final/ directory, not from src/
cd ITAI2373-NewsBot-Final
python -m uvicorn web.api:app
```

**`kagglehub` download fails**
```bash
pip install kagglehub --upgrade
# Or manually download from https://www.kaggle.com/datasets/hgultekin/bbcnewsarchive
# and place CSV in data/raw/
```

**spaCy model not found**
```bash
python -m spacy download en_core_web_sm
```

**NLTK data missing**
```python
import nltk
for pkg in ['punkt', 'stopwords', 'wordnet', 'vader_lexicon', 'punkt_tab']:
    nltk.download(pkg)
```
