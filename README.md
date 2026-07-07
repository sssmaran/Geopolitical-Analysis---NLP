# NLP-HashLLM — Multilingual Policy NLP System

A Streamlit application for multilingual policy discourse analysis, macro‑financial forecasting, and comparative abstractive summarization.

## Overview
- Trend Analysis: Country‑level temporal sentiment aggregation, statement‑level stance classification (XLM‑R), and India news entity/topic analytics (NER, LDA, sentiment).
- Predictive Analysis: Exchange‑rate forecasting (INR/USD, CNY/USD, MXN/USD) via gradient‑boosted regression with lag features, moving averages, RSI/MACD, and tariff‑news sentiment/count signals.
- Summarization: Comparative inference using mT5_XLSum and Pegasus with ROUGE/BLEU evaluation and rationale highlighting via semantic similarity.
- Technology Stack: `Streamlit`, `Transformers`, `XGBoost`, `scikit‑learn`, `spaCy`, `NLTK`, `sentence‑transformers`.

## Project Structure
- `code_files/app.py`: Entrypoint with sidebar navigation and module dispatch.
- `code_files/sections/`:
  - `trend_analysis.py`: Reddit sentiment time series, XLM‑R stance classifier, India NER/topic/sentiment analytics.
  - `predictive_analysis.py`: FX forecasting pipelines for India, China, Mexico.
  - `summarization.py`: mT5_XLSum vs Pegasus summarization, ROUGE/BLEU evaluation, rationale extraction.
- `code_files/stance_xlmr_model/`: Local XLM‑R classifier artifacts (tokenizer and weights).
- `code_files/data/`: Tabular datasets (news corpora, FX series, indices, labeled sentiment, topic labels).
- Notebooks/reports: `*.ipynb`, `*.pdf` (exploration and methodology).

## Requirements
- Python 3.10+
- Optional GPU (CUDA) for accelerated Transformer inference.
- Install dependencies:
  ```bash
  pip install -r code_files/requirements.txt
  ```
- If spaCy model installation fails:
  ```bash
  python -m spacy download en_core_web_sm
  ```
- NLTK resources (`punkt`, `stopwords`, `wordnet`) download automatically on first run.

## Usage
```bash
streamlit run code_files/app.py
```
- Open the local URL printed by Streamlit.
- Navigate via sidebar: Introduction, Trend Analysis, Predictive Analysis, Summarization.

## Capabilities
- Trend Analysis
  - Country‑wise Reddit sentiment timeline from `code_files/data/reddit_sentiment_labeled.csv`.
  - Stance detection (Public vs Government) using the XLM‑R classifier.
  - India news analytics: entity normalization, topic labeling, sentiment scoring, weekly topic trajectories.
- Predictive Analysis
  - FX forecasting with lagged features, moving averages, RSI/MACD, and tariff‑news exogenous signals.
  - Diagnostics: RMSE and R²; visual overlays of actual vs predicted.
- Summarization
  - mT5_XLSum and Pegasus inference.
  - ROUGE/BLEU evaluation against optional references.
  - Rationale extraction via semantic similarity over input sentences.

## Key Data
- `code_files/data/merged_exchange_tariff_data.csv`: INR/USD merged with tariff sentiment and article counts.
- `code_files/data/CNY_USD_exchange.csv`, `code_files/data/MXN_USD_exchange.csv`: FX close series.
- `code_files/data/china_news_with_sentiment.csv`, `code_files/data/mexico_news_with_sentiment.csv`: Tokenized news with sentiment scores.
- `code_files/data/India_sector_news_articles.csv`: Source corpus for India topic/NER enrichment.
- `code_files/data/new_NER.csv`: Materialized India enriched dataset (created on first run if absent).

## Architecture
- Routing: Sidebar `section` dispatches to modules in `code_files/sections/`.
- Caching: Heavy objects cached via `@st.cache_resource` to reduce initialization cost.
- Modeling:
  - Stance: XLM‑R sequence classifier (local weights), CPU/GPU adaptive inference.
  - FX: `XGBRegressor` with log transforms and technical indicators (RSI/MACD), chronological train/test split.
  - Summarization: `google/pegasus-xsum` and `csebuetnlp/mT5_multilingual_XLSum` via `transformers`.

## Operational Notes
- Ensure `code_files/data/` contains required CSVs; update paths in `sections/*` if reorganized.
- Initial runs may incur model and resource downloads.
- For reproducibility, use a virtual environment and pin package versions.

## Launch Checklist
- Dependencies installed successfully.
- `code_files/stance_xlmr_model/` present and readable.
- Required CSVs available under `code_files/data/`.
- Start via `streamlit run code_files/app.py`; navigate via sidebar.
