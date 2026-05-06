# NewsBot Intelligence Engine 2.0 — Deployment Guide

## Option 1: Google Colab (Recommended for Grading)

**Zero setup required.**

1. Open [NewsBot_Intelligence_Engine_2.0.ipynb](../NewsBot_Intelligence_Engine_2.0.ipynb) in Colab
2. Runtime → Run all
3. Pipeline runs end-to-end in ~8 minutes on Colab free tier (CPU)
4. Gradio interface launches with a public share URL at the final cell

**Notes:**
- No API keys required — BBC dataset downloads automatically via `kagglehub`
- `share=True` generates a public tunnel URL valid for 72 hours
- GPU runtime reduces summarization time from ~90s to ~20s

---

## Option 2: Local Python Environment

```bash
git clone https://github.com/icheftech/NewsBot-Intelligence-Engine.git
cd NewsBot-Intelligence-Engine/ITAI2373-NewsBot-Final

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Set Kaggle credentials (for dataset download)
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

jupyter notebook NewsBot_Intelligence_Engine_2.0.ipynb
```

---

## Option 3: Vercel Web App (Bonus — Already Deployed)

The Next.js dashboard is deployed at the URL in the root README.

To redeploy from source:

```bash
cd newsbot-dashboard
npm install
npm run build
vercel --prod
```

**Environment variables** (set in Vercel dashboard):
- No backend secrets required — the dashboard uses static mock responses for the demo

---

## Running Tests

```bash
cd ITAI2373-NewsBot-Final
pip install pytest
pytest tests/ -v --tb=short
```

Expected output: all tests pass in ~30 seconds (no GPU or internet required).

---

## Kaggle Dataset Access

The BBC News Archive dataset downloads automatically via `kagglehub`:

```python
import kagglehub
path = kagglehub.dataset_download('hgultekin/bbcnewsarchive')
```

If you encounter authentication errors, create `~/.kaggle/kaggle.json`:
```json
{"username": "your_username", "key": "your_api_key"}
```

API keys are available at: https://www.kaggle.com/settings → API → Create New Token
