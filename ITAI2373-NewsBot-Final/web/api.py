"""
NewsBot 2.0 — FastAPI Backend
ITAI 2373 Final Project | Leroy Brown | Houston Community College

Run from ITAI2373-NewsBot-Final/ directory:
    uvicorn web.api:app --reload --port 8000

Or directly:
    python web/api.py
"""

import sys
import os

# Ensure src/ is importable when run from any directory
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── NLP pipeline imports ───────────────────────────────────────────────────
import kagglehub
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from config.settings import (
    DATASET_SIZE, RANDOM_STATE, N_TOPICS, EMBEDDING_MODEL, SUMMARIZER_MODEL
)
from src.data_processing.data_loader import load_bbc_dataset
from src.data_processing.data_validator import validate_dataframe, clean_dataframe
from src.data_processing.preprocessor import preprocess_text
from src.analysis.classifier import NewsClassifier
from src.analysis.topic_modeler import TopicModeler
from src.analysis.sentiment_analyzer import score_text
from src.language_models.summarizer import summarize_with_stats
from src.language_models.embeddings import SemanticSearchIndex
from src.multilingual.translator import translate_and_detect
from src.conversation.query_processor import QueryProcessor
from src.conversation.intent_classifier import classify_intent

# ── Global pipeline state ──────────────────────────────────────────────────
pipeline = {}


def boot_pipeline():
    """Load dataset, fit all models, and store in global pipeline dict."""
    print("Booting NewsBot 2.0 pipeline...")

    # 1. Dataset
    path = kagglehub.dataset_download("hgultekin/bbcnewsarchive")
    df_raw = load_bbc_dataset(path)
    df = clean_dataframe(df_raw).sample(
        min(DATASET_SIZE, len(df_raw)), random_state=RANDOM_STATE
    ).reset_index(drop=True)
    df["clean_text"] = df["text"].apply(preprocess_text)
    df["word_count"] = df["text"].apply(lambda t: len(t.split()))
    print(f"  Dataset: {len(df):,} articles")

    # 2. TF-IDF + Classifier
    vectorizer = TfidfVectorizer(max_features=10_000, ngram_range=(1, 2), min_df=2)
    X_tfidf = vectorizer.fit_transform(df["clean_text"])
    clf = NewsClassifier(vectorizer=vectorizer)
    results = clf.train_test_evaluate(X_tfidf, df["category"])
    print(f"  Classifier: {results['accuracy']:.1%} accuracy")

    # 3. Sentiment
    sentiments = df["text"].apply(score_text)
    df["compound"] = sentiments.apply(lambda s: s["sent_compound"])
    print("  Sentiment: scored")

    # 4. Topic modeling (LDA)
    count_vec = CountVectorizer(max_features=5_000, min_df=2, stop_words="english")
    dtm = count_vec.fit_transform(df["clean_text"])
    lda = TopicModeler(n_topics=N_TOPICS, method="lda")
    lda.fit_transform(dtm, count_vec)
    top_topics = {
        cat: lda.get_topic_words(i % N_TOPICS, n_words=8)
        for i, cat in enumerate(df["category"].unique())
    }
    print("  Topics: LDA fitted")

    # 5. Semantic search
    search_index = SemanticSearchIndex()
    search_index.build(df)
    print(f"  Search index: {len(search_index.index)} articles")

    # 6. Stats cache
    stats_cache = {
        "total_articles":  len(df),
        "accuracy":        results["accuracy"],
        "category_counts": df["category"].value_counts().to_dict(),
        "avg_sentiment":   df.groupby("category")["compound"].mean().round(4).to_dict(),
        "avg_words":       df.groupby("category")["word_count"].mean().round(0).astype(int).to_dict(),
        "top_topics":      top_topics,
    }

    # 7. Query processor
    qp = QueryProcessor(
        classifier=clf,
        vectorizer=vectorizer,
        preprocessor=preprocess_text,
        summarizer_fn=summarize_with_stats,
        search_index=search_index,
        translator_fn=translate_and_detect,
        sentiment_fn=score_text,
        stats_cache=stats_cache,
    )

    pipeline.update({
        "df": df,
        "clf": clf,
        "vectorizer": vectorizer,
        "search_index": search_index,
        "stats_cache": stats_cache,
        "qp": qp,
        "ready": True,
    })
    print("Pipeline ready.\n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    boot_pipeline()
    yield


# ── FastAPI app ────────────────────────────────────────────────────────────
app = FastAPI(
    title="NewsBot 2.0 API",
    description="AI-powered news analysis — ITAI 2373 Final Project",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── Request / Response models ──────────────────────────────────────────────
class TextRequest(BaseModel):
    text: str

class QueryRequest(BaseModel):
    message: str
    history: Optional[list] = []

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5


# ── Health ────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok", "ready": pipeline.get("ready", False)}


# ── Chat (routes through QueryProcessor) ─────────────────────────────────
@app.post("/api/chat")
def chat(req: QueryRequest):
    if not pipeline.get("ready"):
        raise HTTPException(503, "Pipeline not ready")
    response = pipeline["qp"].process(req.message, req.history)
    intent = classify_intent(req.message)
    return {"response": response, "intent": intent}


# ── Classify ──────────────────────────────────────────────────────────────
@app.post("/api/classify")
def classify(req: TextRequest):
    if not pipeline.get("ready"):
        raise HTTPException(503, "Pipeline not ready")
    if len(req.text.split()) < 10:
        raise HTTPException(400, "Text too short — minimum 10 words")

    clf = pipeline["clf"]
    vectorizer = pipeline["vectorizer"]
    clean = preprocess_text(req.text)
    X = vectorizer.transform([clean])
    category = clf.model.predict(X)[0]
    probs = clf.model.predict_proba(X)[0]
    proba_pairs = sorted(
        zip(clf.model.classes_, probs.tolist()), key=lambda x: -x[1]
    )
    sentiment = score_text(req.text)
    return {
        "category":    category,
        "confidence":  round(proba_pairs[0][1], 4),
        "all_scores":  [{"category": c, "probability": round(p, 4)} for c, p in proba_pairs],
        "sentiment":   sentiment,
    }


# ── Summarize ─────────────────────────────────────────────────────────────
@app.post("/api/summarize")
def summarize(req: TextRequest):
    if not pipeline.get("ready"):
        raise HTTPException(503, "Pipeline not ready")
    if len(req.text.split()) < 40:
        raise HTTPException(400, "Text too short — minimum 40 words")
    result = summarize_with_stats(req.text)
    return result


# ── Search ────────────────────────────────────────────────────────────────
@app.post("/api/search")
def search(req: SearchRequest):
    if not pipeline.get("ready"):
        raise HTTPException(503, "Pipeline not ready")
    if len(req.query.strip()) < 3:
        raise HTTPException(400, "Query too short")
    results = pipeline["search_index"].search(req.query, top_k=req.top_k)
    return {"query": req.query, "results": results}


# ── Translate ─────────────────────────────────────────────────────────────
@app.post("/api/translate")
def translate(req: TextRequest):
    if not pipeline.get("ready"):
        raise HTTPException(503, "Pipeline not ready")
    result = translate_and_detect(req.text)
    if result.get("error"):
        raise HTTPException(500, result["error"])

    # Classify translated text
    if result["source_lang"] not in ("en", "unknown"):
        clf = pipeline["clf"]
        vectorizer = pipeline["vectorizer"]
        clean = preprocess_text(result["translated"])
        X = vectorizer.transform([clean])
        result["predicted_category"] = clf.model.predict(X)[0]
        result["sentiment"] = score_text(result["translated"])

    return result


# ── Stats ─────────────────────────────────────────────────────────────────
@app.get("/api/stats")
def stats():
    if not pipeline.get("ready"):
        raise HTTPException(503, "Pipeline not ready")
    return pipeline["stats_cache"]


# ── Topics ────────────────────────────────────────────────────────────────
@app.get("/api/topics")
def topics():
    if not pipeline.get("ready"):
        raise HTTPException(503, "Pipeline not ready")
    return pipeline["stats_cache"].get("top_topics", {})


# ── Serve frontend ────────────────────────────────────────────────────────
@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


if __name__ == "__main__":
    uvicorn.run("web.api:app", host="0.0.0.0", port=8000, reload=False)
