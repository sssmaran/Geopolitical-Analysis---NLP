# NLP-HashLLM — Geopolitical Policy Analysis - NLP Application

Analyze public and government discourse around tariffs and trade policy, forecast economic indicators, and compare multilingual summarizers — all in one Streamlit app.

## Overview
- Interactive sections for exploration and modeling:
  - Trend Analysis: Temporal sentiment by country, stance detection for statements, and India-focused topic trends.
  - Predictive Analysis: XGBoost-based forecasts for INR/USD, CNY/USD, and MXN/USD using market features and tariff-news signals.
  - Summarization: Side-by-side comparison of mT5_XLSum and Pegasus with ROUGE/BLEU evaluation and rationale highlighting.
- Built with `streamlit`, `transformers`, `xgboost`, `scikit-learn`, `spacy`, `nltk`, and `sentence-transformers`.

## Project Structure
- `code_files/app.py`: Streamlit entrypoint with sidebar navigation and section routing.
- `code_files/sections/`:
  - `trend_analysis.py`: Reddit sentiment trends, XLM-R stance classifier, India NER/topic/sentiment analytics.
  - `predictive_analysis.py`: Exchange-rate forecasting for India, China, Mexico.
  - `summarization.py`: mT5_XLSum vs Pegasus summaries with ROUGE/BLEU and rationale.
- `code_files/stance_xlmr_model/`: Pretrained XLM-R sequence classifier (tokenizer + weights) for stance detection.
- `code_files/data/`: CSV datasets used across sections (news, FX, indices, labeled sentiment, topic labels, etc.).
- Notebooks and reports: `NLP_Final.ipynb`, `NLP_Milestone.ipynb`, `NLP_Final_Report.pdf`, `NLP_Report_2 (1).pdf` for methodology and experiments.

## Requirements
- Python 3.10+
- Recommended: CUDA-enabled GPU for faster Transformer inference (CPU works).
- Install dependencies:
  ```bash
  pip install -r code_files/requirements.txt
  ```
- spaCy small English model is specified in `requirements.txt`. If installation fails, manually run:
  ```bash
  python -m spacy download en_core_web_sm
  ```
- NLTK resources (`punkt`, `stopwords`, `wordnet`) are downloaded automatically on first run when generating enriched NER/topic data.

## Run the App
```bash
streamlit run code_files/app.py
```
- Open the browser link Streamlit prints.
- Use the sidebar to switch sections: Introduction, Trend Analysis, Predictive Analysis, Summarization.

## Features
- Trend Analysis
  - Country-wise Reddit sentiment timeline from `code_files/data/reddit_sentiment_labeled.csv`.
  - Stance detection (Public vs Government) using the bundled XLM-R model in `code_files/stance_xlmr_model/`.
  - India news analytics: cleaned NER entities, economic term extraction, LDA topic labels, sentiment, and weekly topic trends.
- Predictive Analysis
  - INR/USD, CNY/USD, MXN/USD forecasting with engineered lags, moving averages, RSI/MACD (TA), and tariff-news signals.
  - Visual comparison of actual vs predicted with RMSE and R² metrics.
- Summarization
  - mT5_XLSum and Pegasus summaries computed on demand.
  - ROUGE and BLEU evaluation against an optional reference.
  - Rationale highlighting: shows input sentences most similar to summary content via `sentence-transformers` embeddings.

## Key Data Files
- `code_files/data/merged_exchange_tariff_data.csv`: INR/USD with tariff sentiment and counts.
- `code_files/data/CNY_USD_exchange.csv`, `code_files/data/MXN_USD_exchange.csv`: FX time series.
- `code_files/data/china_news_with_sentiment.csv`, `code_files/data/mexico_news_with_sentiment.csv`: News with sentiment.
- `code_files/data/India_sector_news_articles.csv`: Source for India topic/NER enrichment.
- `code_files/data/new_NER.csv`: Generated India enriched dataset (created on first run if missing).

## How It Works
- App Routing: Sidebar sets `section` and dispatches to analysis modules from `code_files/sections/`.
- Caching: Heavy models and embeddings are cached using `@st.cache_resource` to avoid repeated downloads and initialization.
- Modeling:
  - Stance classifier uses XLM-R loaded locally; inference runs on GPU if available, else CPU.
  - FX forecasting uses `XGBRegressor` with log transforms and technical indicators.
  - Summarizers use `google/pegasus-xsum` and `csebuetnlp/mT5_multilingual_XLSum` via `transformers`.

## Tips
- If large datasets are missing, ensure `code_files/data/` contains the listed CSVs.
- First run may take longer due to model and resource downloads.
- For reproducible results, pin Python and package versions and run in a fresh virtual environment.

## Launch Checklist
- Dependencies installed without errors.
- `code_files/stance_xlmr_model/` present and readable.
- CSVs in `code_files/data/` present (or adjust paths inside `sections/*`).
- Start with `streamlit run code_files/app.py` and navigate via sidebar.

