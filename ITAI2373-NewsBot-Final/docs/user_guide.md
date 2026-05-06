# NewsBot Intelligence Engine 2.0 — User Guide

## Quick Start

Open the notebook in Google Colab:

1. Navigate to `ITAI2373-NewsBot-Final/NewsBot_Intelligence_Engine_2.0.ipynb`
2. Click the **Open in Colab** badge in the README
3. Select **Runtime → Run all**
4. Wait ~8 minutes for the full pipeline to initialize
5. The Gradio chat interface launches automatically at the end with a public URL

---

## Using the Gradio Chat Interface

### Available Commands

| Command | Format | Example |
|---------|--------|---------|
| Classify | `classify: [article text]` | `classify: The Prime Minister announced new fiscal policies today.` |
| Summarize | `summarize: [article text]` | `summarize: [paste a 100+ word article]` |
| Search | `search: [topic]` | `search: Premier League transfer news` |
| Translate | `translate: [foreign text]` | `translate: El gobierno anunció nuevas medidas.` |
| Stats | `stats` | `stats` |
| Topics | `topics` | `topics` |
| Help | `help` | `help` |

### Tips
- The **classify** command requires at least 20 words of article text for a reliable prediction
- The **summarize** command requires at least 40 words
- **Search** uses semantic similarity — natural language questions work better than keywords alone
- **Translate** supports any language detected by Google Translate (50+ languages)

---

## Using the Web Dashboard

The production Next.js dashboard is deployed at the Vercel URL listed in the README.

**Pages:**
- **Dashboard** — System overview and quick stats
- **Classify** — Paste an article and get category + confidence chart
- **Summarize** — Get compressed summaries with word count stats
- **Search** — Semantic article search with relevance scores
- **Translate** — Translate and auto-classify non-English articles

---

## Running Tests

```bash
cd ITAI2373-NewsBot-Final
pip install pytest
pytest tests/ -v
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run Cell 1 to install all dependencies |
| Gradio URL not generated | Ensure `share=True` in the final cell's `demo.launch()` call |
| Slow summarization | DistilBART runs on CPU by default — expected ~90s for 5 articles |
| `LangDetectException` | Short texts (<10 words) may fail language detection — provide more text |
| Kaggle dataset download fails | Set `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables |
