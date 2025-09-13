# Introducing FinSight- a Stock Market Sentiment Analysis Dashboard 

A comprehensive Streamlit dashboard that combines financial news sentiment analysis with stock return correlations using VADER and FinBERT models.

## Features

- **Real-time Stock Data**: Fetches live stock prices via Yahoo Finance
- **Dual Sentiment Analysis**: Uses both VADER and FinBERT for comprehensive sentiment scoring
- **Interactive Visualizations**: Plotly-powered charts showing sentiment vs returns
- **Market Overview**: Aggregate sentiment trends across multiple stocks
- **Custom Analysis**: Analyze any financial headline in real-time

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/RahemShurjeel83/FinSight.git
   cd stock-sentiment-dashboard
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment:**
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

Run the Streamlit dashboard:
```bash
streamlit run FinSight.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 📊 How to Use

1. **Select Stocks**: Choose from popular stocks or enter custom ticker symbols
2. **Analyze Headlines**: Enter financial news headlines for sentiment analysis
3. **View Correlations**: Explore how sentiment correlates with stock returns
4. **Market Overview**: Get aggregate sentiment trends across multiple stocks
5. **Real-time Updates**: Refresh for latest stock prices and sentiment scores


## 🔧 Technologies Used

- **Streamlit** - Web dashboard framework
- **VADER Sentiment** - Rule-based sentiment analysis
- **FinBERT** - Financial domain-specific BERT model
- **Yahoo Finance API** - Real-time stock data
- **Plotly** - Interactive visualizations
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing

## 📈 Models & Analysis

- **VADER**: Lexicon-based sentiment analyzer optimized for social media text
- **FinBERT**: Transformer model fine-tuned on financial text for domain-specific sentiment
- **Correlation Analysis**: Statistical analysis between sentiment scores and stock returns
- **Visualization**: Interactive charts showing sentiment trends and market movements


## 📝 License

This project is open source and available under the [MIT License](LICENSE).

