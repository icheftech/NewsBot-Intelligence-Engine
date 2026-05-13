# NewsBot 2.0 — Web Application

FastAPI backend + HTML/JS single-page frontend for the NewsBot Intelligence Engine 2.0.

---

## Requirements

Install additional dependencies (on top of the main `requirements.txt`):

```bash
pip install fastapi uvicorn python-multipart
```

---

## Running the App

From the **`ITAI2373-NewsBot-Final/`** directory:

```bash
uvicorn web.api:app --reload --port 8000
```

Then open your browser to: **http://localhost:8000**

> The first launch takes 1–2 minutes while the NLP pipeline initializes (downloads dataset, fits TF-IDF, builds semantic index). The UI displays a loading indicator until the backend is ready.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Pipeline status check |
| POST | `/api/chat` | Conversational interface (QueryProcessor) |
| POST | `/api/classify` | Classify article text (category + confidence + sentiment) |
| POST | `/api/summarize` | Abstractive summarization (DistilBART) |
| POST | `/api/search` | Semantic search (Sentence-BERT cosine similarity) |
| POST | `/api/translate` | Language detect + translate + classify |
| GET | `/api/stats` | Dataset analytics and model performance |
| GET | `/api/topics` | LDA topic keywords per category |

### Example: Classify

```bash
curl -X POST http://localhost:8000/api/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Apple unveiled its latest iPhone at a special event Tuesday..."}'
```

### Example: Chat

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What topics are in the tech category?", "history": []}'
```

---

## Frontend Features

- **Chat** — Full conversational interface with 7 intents (classify, summarize, search, translate, stats, topics, help)
- **Classify** — Paste article text → get category, confidence bar chart, and sentiment
- **Summarize** — Paste article → get abstractive summary with compression stats
- **Search** — Enter topic → semantic search returns top matching articles
- **Translate** — Paste foreign-language text → detect language, translate, classify

---

## Running in Google Colab (Alternative)

```python
!pip install fastapi uvicorn python-multipart nest-asyncio pyngrok

import nest_asyncio
from pyngrok import ngrok
nest_asyncio.apply()

ngrok.set_auth_token("YOUR_NGROK_TOKEN")
public_url = ngrok.connect(8000)
print(f"Public URL: {public_url}")

import uvicorn
uvicorn.run("web.api:app", host="0.0.0.0", port=8000)
```

---

## Project Structure

```
web/
├── api.py          # FastAPI backend — pipeline + REST endpoints
├── static/
│   └── index.html  # Single-page frontend (Tailwind CSS + vanilla JS)
└── README.md       # This file
```
