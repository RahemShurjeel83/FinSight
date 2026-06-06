# FinSight — Stock Market Sentiment Analysis Dashboard

A Streamlit dashboard that combines financial news sentiment analysis with stock return correlations using VADER and FinBERT models.

## Features

- **Real-time Stock Data** — fetches live prices via Yahoo Finance
- **Dual Sentiment Analysis** — VADER (rule-based) + FinBERT (domain-specific transformer)
- **Interactive Visualizations** — Plotly charts overlaying sentiment scores with stock returns
- **Market Overview** — aggregate sentiment trends across multiple tickers
- **Custom Analysis** — score any financial headline on the fly

## Dataset

The app uses the [FinancialPhraseBank](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news) dataset (`all-data.csv`).  
Download it from Kaggle and place it in the same folder as `FinSight.py` before running.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/RahemShurjeel83/FinSight.git
   cd FinSight
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Mac / Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

```bash
streamlit run FinSight.py
```

The dashboard opens in your browser at `http://localhost:8501`.

## How to Use

1. **Home** — pick a ticker, set a date range, and click *Run Analysis* to see sentiment vs. returns
2. **Custom Analysis** — enter any financial headline and get VADER + FinBERT scores instantly
3. **Market Overview** — explore aggregate sentiment trends and score distributions across all tickers

## Technologies

| Library | Purpose |
|---|---|
| Streamlit | Web dashboard |
| VADER Sentiment | Rule-based sentiment analysis |
| FinBERT (ProsusAI) | Finance-domain transformer model |
| yfinance | Real-time stock data |
| Plotly | Interactive charts |
| Pandas / NumPy | Data processing |
| PyTorch | FinBERT backend |

## Models

- **VADER** — lexicon-based analyzer, fast and good for short financial phrases
- **FinBERT** — BERT fine-tuned on financial text; slower but domain-aware
- Scores range from **-1** (very negative) to **+1** (very positive)

## License

MIT License — see [LICENSE](LICENSE).
