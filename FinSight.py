import os
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Data Ingestion

@st.cache_data
def load_financial_phrasebank():
    """
    Load FinancialPhraseBank dataset from Kaggle.
    Expected columns in all-data.csv: ['sentence', 'sentiment']
    We'll need to transform this for our use case.
    """
    try:
        
        df = pd.read_csv("all-data.csv", encoding='latin-1', header=None)
        df.columns = ['sentence', 'sentiment']
        num_rows = len(df)
        date_range = pd.date_range(end=datetime.now(), periods=min(num_rows, 180))
        
        if num_rows > len(date_range):
            dates = np.random.choice(date_range, size=num_rows, replace=True)
        else:
            dates = date_range[:num_rows]
        
        df['date'] = pd.to_datetime(dates)
        df['title'] = df['sentence']  
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'BAC', 'WMT']
        df['ticker'] = np.random.choice(tickers, size=len(df))
        
        return df[['date', 'title', 'ticker', 'sentiment']]
    except FileNotFoundError:
        st.error("❌ Could not find 'all-data.csv'. Please ensure it's in the same directory as app.py")
        return pd.DataFrame(columns=["date", "title", "ticker", "sentiment"])
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame(columns=["date", "title", "ticker", "sentiment"])

# Sentiment Analysis 

@st.cache_resource
def init_sentiment_models():
    """Initialize sentiment analysis models."""
    try:
        vader = SentimentIntensityAnalyzer()
        finbert = pipeline(
            "sentiment-analysis", 
            model="ProsusAI/finbert",
            device=-1,
            return_all_scores=True,  
            truncation=True,
            max_length=512
        )
        return vader, finbert
    except Exception as e:
        st.error(f"Error initializing models: {e}")
        st.info("Make sure you have installed transformers: pip install transformers")
        return None, None

def analyze_sentiment(text, vader, finbert):
    """Analyze sentiment using both VADER and FinBERT - FIXED VERSION."""
    vader_score = 0
    finbert_score = 0
    
    try:
       
        vader_result = vader.polarity_scores(text)
        vader_score = vader_result["compound"]
        
        text_truncated = text[:512] if len(text) > 512 else text
        finbert_result = finbert(text_truncated)
        if st.session_state.get('debug_finbert', False):
            st.write(f"Raw FinBERT result: {finbert_result}")
        
        if finbert_result and len(finbert_result) > 0:
            scores = finbert_result[0]  
            
           
            score_map = {}
            for item in scores:
                score_map[item['label'].lower()] = item['score']
            
            positive_score = score_map.get('positive', 0)
            negative_score = score_map.get('negative', 0)
            neutral_score = score_map.get('neutral', 0)
            finbert_score = positive_score - negative_score
            
        else:
            st.warning(f"No FinBERT result for text: {text[:50]}...")
            
    except Exception as e:
        error_msg = f"Error in sentiment analysis for text '{text[:50]}...': {str(e)}"
        if st.session_state.get('debug_finbert', False):
            st.error(error_msg)
        else:
            print(error_msg)
    
    return vader_score, finbert_score


@st.cache_data
def calculate_sentiment_scores(news_df, _vader, _finbert):
    """Calculate sentiment scores for all headlines (cached to prevent recalculation)."""
    if news_df.empty:
        return news_df
    
    if "vader_score" not in news_df.columns or "finbert_score" not in news_df.columns:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        scores = []
        failed_count = 0
        
        for idx, row in news_df.iterrows():
            v_score, f_score = analyze_sentiment(str(row["title"]), _vader, _finbert)
            
            if f_score == 0:
                failed_count += 1
            
            scores.append([v_score, f_score])
            progress = (idx + 1) / len(news_df)
            progress_bar.progress(progress)
            status_text.text(f'Processing headline {idx + 1}/{len(news_df)} (FinBERT failures: {failed_count})')
        
        progress_bar.empty()
        status_text.empty()
        
        if failed_count > 0:
            st.warning(f"⚠️ FinBERT failed to score {failed_count}/{len(news_df)} headlines. This may explain zero scores.")
        
        sentiment_df = pd.DataFrame(scores, columns=["vader_score", "finbert_score"])
        news_df = pd.concat([news_df.reset_index(drop=True), sentiment_df], axis=1)
    
    return news_df


# Stock Data

@st.cache_data
def fetch_stock_data(ticker, start, end):
    """Fetch stock data from Yahoo Finance."""
    try:
        
        ticker = ticker.upper().strip()
        stock = yf.download(ticker, start=start, end=end, progress=False)
        
        if stock.empty:
            st.warning(f"No data found for ticker {ticker}. Please check if the ticker symbol is correct.")
            return pd.DataFrame()
        
        required_columns = ['Close']
        if not all(col in stock.columns or any(col in str(c) for c in stock.columns) for col in required_columns):
            st.error(f"Missing required price data columns for {ticker}")
            return pd.DataFrame()
            
        return stock
    except Exception as e:
        st.error(f"Error fetching stock data for {ticker}: {e}")
        return pd.DataFrame()


# Correlation Engine 

def correlate_sentiment_with_returns(news_df, stock_df, ticker):
    
    if news_df.empty or stock_df.empty:
        return pd.DataFrame()
    
    try:
        ticker_news = news_df[news_df['ticker'].str.upper() == ticker.upper()] if 'ticker' in news_df.columns else news_df
        
        if ticker_news.empty:
            ticker_news = news_df
            st.warning(f"No ticker-specific news found for {ticker}. Using all available news data.")

        ticker_news = ticker_news.dropna(subset=['vader_score', 'finbert_score'])
        
        if ticker_news.empty:
            st.error("All sentiment scores are NaN or missing!")
            return pd.DataFrame()

        daily_sentiment = (
            ticker_news.groupby("date")[["vader_score", "finbert_score"]]
            .agg(['mean', 'count'])  
            .reset_index()
        )
        
        daily_sentiment.columns = ['date', 'vader_score', 'finbert_score', 'vader_count', 'finbert_count']
        
        daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date']).dt.normalize()

        stock = stock_df.copy()
        stock = stock.reset_index()
        
        if isinstance(stock.columns, pd.MultiIndex):
            
            stock.columns = ['_'.join(col).strip() if isinstance(col, tuple) else str(col) for col in stock.columns]
           
            stock.columns = [col.replace('_', '').replace(' ', '') if col.endswith('_') else col for col in stock.columns]
        
       
        date_col = None
        possible_date_cols = ['Date', 'date', 'Datetime', 'datetime', 'index']
        for col in possible_date_cols:
            if col in stock.columns:
                date_col = col
                break
        
        if date_col is None:
            
            for col in stock.columns:
                if pd.api.types.is_datetime64_any_dtype(stock[col]):
                    date_col = col
                    break
        
        if date_col is None:
            st.error("Could not find date column in stock data")
            return pd.DataFrame()

        close_col = None
        possible_close_cols = ['Close', 'close', 'AdjClose', 'Adj Close', 'adj_close']
        for col in possible_close_cols:
            if col in stock.columns:
                close_col = col
                break
        
        if close_col is None:
            for col in stock.columns:
                if 'close' in col.lower():
                    close_col = col
                    break
        
        if close_col is None:
            st.error(f"Could not find Close price column in stock data. Available columns: {list(stock.columns)}")
            return pd.DataFrame()

        stock['Date'] = pd.to_datetime(stock[date_col]).dt.normalize()
        stock = stock.dropna(subset=['Date', close_col])
        
       
        stock['return'] = stock[close_col].pct_change() * 100
        stock['Close'] = stock[close_col]  
        
        merged = pd.merge(
            daily_sentiment,
            stock[['Date', 'Close', 'return']],
            left_on="date",
            right_on="Date",
            how="inner"
        )
        
        merged = merged.dropna()
        
        if merged.empty:
            st.warning(f"No overlapping dates found between sentiment data and stock data for {ticker}")
            st.info("Sentiment data range: {} to {}".format(
                daily_sentiment['date'].min().strftime('%Y-%m-%d'),
                daily_sentiment['date'].max().strftime('%Y-%m-%d')
            ))
            st.info("Stock data range: {} to {}".format(
                stock['Date'].min().strftime('%Y-%m-%d'),
                stock['Date'].max().strftime('%Y-%m-%d')
            ))
        
        return merged

    except Exception as e:
        st.error(f"Error correlating data for {ticker}: {e}")
        st.error("Debug info:")
        st.write("Stock columns:", list(stock_df.columns) if not stock_df.empty else "Empty DataFrame")
        st.write("News columns:", list(news_df.columns) if not news_df.empty else "Empty DataFrame")
        return pd.DataFrame()



# Dashboard Layout 


def display_analysis_results(merged, ticker):
    """Display sentiment vs stock returns visualization."""
    st.subheader(f"📊 Sentiment vs Stock Returns for {ticker}")
    
    if merged.empty:
        st.warning("No merged data available to display.")
        return
    
    nan_check = merged[['vader_score', 'finbert_score', 'return']].isna().sum()
    if nan_check.any():
        st.warning("Found NaN values in data:")
        st.write(nan_check)
        # Remove NaN rows
        merged_clean = merged.dropna(subset=['vader_score', 'finbert_score', 'return'])
        st.info(f"Using {len(merged_clean)}/{len(merged)} rows after removing NaN values")
        merged = merged_clean
    
    if merged.empty:
        st.error("No valid data remaining after cleaning NaN values")
        return
    
   
    fig = go.Figure()
    
    
    fig.add_trace(go.Bar(
        x=merged["date"], 
        y=merged["finbert_score"], 
        name="FinBERT Sentiment",
        marker_color='lightblue',
        yaxis='y',
        opacity=0.7
    ))
    
    fig.add_trace(go.Bar(
        x=merged["date"], 
        y=merged["vader_score"], 
        name="VADER Sentiment",
        marker_color='lightgreen',
        yaxis='y',
        opacity=0.7
    ))
    
   
    fig.add_trace(go.Scatter(
        x=merged["date"], 
        y=merged["return"], 
        name="Stock Returns (%)",
        line=dict(color='red', width=3),
        yaxis='y2'
    ))
    
    
    fig.update_layout(
        title=f"{ticker} Sentiment & Returns Analysis",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Sentiment Score", side='left'),
        yaxis2=dict(title="Stock Returns (%)", overlaying='y', side='right'),
        legend=dict(x=0, y=1.1, orientation="h"),
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    
    if len(merged) > 1:
        col1, col2, col3, col4 = st.columns(4)
        
        
        try:
            corr_vader = merged[['vader_score', 'return']].corr().iloc[0, 1]
            corr_finbert = merged[['finbert_score', 'return']].corr().iloc[0, 1]
        except:
            corr_vader = 0
            corr_finbert = 0
        
        with col1:
            st.metric("VADER-Return Correlation", f"{corr_vader:.3f}")
        with col2:
            st.metric("FinBERT-Return Correlation", f"{corr_finbert:.3f}")
        with col3:
            avg_return = merged['return'].mean()
            st.metric("Avg Daily Return", f"{avg_return:.2f}%")
        with col4:
            data_points = len(merged)
            st.metric("Data Points", f"{data_points}")
    
    
    with st.expander("Sentiment Score Statistics"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**VADER Scores:**")
            st.write(f"Mean: {merged['vader_score'].mean():.3f}")
            st.write(f"Std: {merged['vader_score'].std():.3f}")
            st.write(f"Min: {merged['vader_score'].min():.3f}")
            st.write(f"Max: {merged['vader_score'].max():.3f}")
        
        with col2:
            st.write("**FinBERT Scores:**")
            st.write(f"Mean: {merged['finbert_score'].mean():.3f}")
            st.write(f"Std: {merged['finbert_score'].std():.3f}")
            st.write(f"Min: {merged['finbert_score'].min():.3f}")
            st.write(f"Max: {merged['finbert_score'].max():.3f}")
    
    
    with st.expander("View Raw Data"):
        st.dataframe(merged)


def test_finbert_output(finbert):
    """Test FinBERT with sample financial texts."""
    test_texts = [
        "The company reported strong quarterly earnings",
        "Stock prices plummeted amid recession fears", 
        "The market remained stable today",
        "Investors are optimistic about future growth",
        "Economic uncertainty continues to impact markets"
    ]
    
    st.write("**Testing FinBERT with sample texts:**")
    
    for text in test_texts:
        try:
            raw_result = finbert(text)
            st.write(f"Text: '{text}'")
            st.write(f"Raw output: {raw_result}")
    
            if raw_result and len(raw_result) > 0:
                scores = raw_result[0]
                score_map = {}
                for item in scores:
                    score_map[item['label'].lower()] = item['score']
                
                positive_score = score_map.get('positive', 0)
                negative_score = score_map.get('negative', 0)
                final_score = positive_score - negative_score
                
                st.write(f"Processed score: {final_score:.3f}")
            st.write("---")
            
        except Exception as e:
            st.error(f"Error processing '{text}': {e}")


def show_custom_analysis_page(vader, finbert):
    """Page for analyzing custom headlines."""
    st.subheader("Custom Headline Sentiment Analysis")
    
    
    debug_mode = st.checkbox("Enable Debug Mode (shows FinBERT raw output)")
    st.session_state['debug_finbert'] = debug_mode
    
    
    with st.expander("Debug FinBERT (Click to troubleshoot zero scores)"):
        if st.button("Test FinBERT Output"):
            test_finbert_output(finbert)
    
    text = st.text_area("Enter a financial news headline:", 
                       value="Apple Inc. reported record quarterly earnings, beating analyst expectations.",
                       height=100)
    
    if st.button("Analyze Sentiment", type="primary"):
        if text.strip():
            with st.spinner("Analyzing sentiment..."):
                v_score, f_score = analyze_sentiment(text, vader, finbert)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("VADER Score", f"{v_score:.3f}")
                    if v_score > 0.05:
                        st.success("Positive sentiment (VADER)")
                    elif v_score < -0.05:
                        st.error("Negative sentiment (VADER)")
                    else:
                        st.info("Neutral sentiment (VADER)")
                
                with col2:
                    st.metric("FinBERT Score", f"{f_score:.3f}")
                    if f_score > 0.05:
                        st.success("Positive sentiment (FinBERT)")
                    elif f_score < -0.05:
                        st.error("Negative sentiment (FinBERT)")
                    else:
                        st.info("Neutral sentiment (FinBERT)")
        else:
            st.warning("Please enter a headline to analyze.")


def show_market_overview_page(news_df):
    """Display market-wide sentiment overview - FIXED."""
    st.subheader("Market-Wide Sentiment Overview")
    
    if news_df.empty:
        st.warning("No news data available.")
        return
    
    
    clean_df = news_df.dropna(subset=['vader_score', 'finbert_score'])
    
    if clean_df.empty:
        st.error("All sentiment scores are NaN! Check the FinBERT model initialization.")
        return
      
    total_rows = len(news_df)
    valid_rows = len(clean_df)
    
    if valid_rows < total_rows:
        st.warning(f"Using {valid_rows}/{total_rows} rows with valid sentiment scores")
     
    daily_avg = (
        clean_df.groupby("date")[["vader_score", "finbert_score"]]
        .mean()
        .reset_index()
    )
       
    fig = px.line(
        daily_avg,
        x="date",
        y=["vader_score", "finbert_score"],
        labels={"value": "Sentiment Score", "variable": "Model", "date": "Date"},
        title="Average Market Sentiment Over Time"
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
     
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_vader = clean_df["vader_score"].mean()
        st.metric("Avg VADER Score", f"{avg_vader:.3f}")
    with col2:
        avg_finbert = clean_df["finbert_score"].mean()
        st.metric("Avg FinBERT Score", f"{avg_finbert:.3f}")
    with col3:
        total_headlines = len(clean_df)
        st.metric("Valid Headlines", f"{total_headlines:,}")
    with col4:
        unique_tickers = clean_df["ticker"].nunique() if "ticker" in clean_df.columns else 0
        st.metric("Unique Tickers", f"{unique_tickers}")
       
    with st.expander("Score Distributions"):
        col1, col2 = st.columns(2)
        
        with col1:
            fig_vader = px.histogram(clean_df, x='vader_score', title='VADER Score Distribution')
            st.plotly_chart(fig_vader, use_container_width=True)
        
        with col2:
            fig_finbert = px.histogram(clean_df, x='finbert_score', title='FinBERT Score Distribution')
            st.plotly_chart(fig_finbert, use_container_width=True)


# Main App 

def main():
    st.set_page_config(
        page_title="FinSight Sentiment Dashboard",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📈 FinSight Dashboard")
    st.markdown("---")
    
    
    with st.spinner("Loading sentiment models..."):
        vader, finbert = init_sentiment_models()
    
    if vader is None or finbert is None:
        st.error("Failed to load sentiment models. Please check your installation.")
        return
    
   
    if 'sentiment_calculated' not in st.session_state:
        st.session_state.sentiment_calculated = False
        st.session_state.news_df = pd.DataFrame()
        st.session_state.debug_finbert = False
    
  
    if st.session_state.news_df.empty:
        news_df = load_financial_phrasebank()
        if not news_df.empty:
            
            with st.spinner("Calculating sentiment scores for all headlines..."):
                news_df = calculate_sentiment_scores(news_df, vader, finbert)
            st.session_state.news_df = news_df
            st.session_state.sentiment_calculated = True
            
            
            if not news_df.empty:
                zero_finbert = (news_df['finbert_score'] == 0).sum()
                total_rows = len(news_df)
                st.info(f"Processed {total_rows} headlines. FinBERT zero scores: {zero_finbert} ({zero_finbert/total_rows*100:.1f}%)")
    else:
        news_df = st.session_state.news_df
    
   
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["Home", "Custom Analysis", "Market Overview"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Note**: Start with Market Overview to see overall sentiment trends, "
        "then analyze specific stocks in the Home page."
    )
        
    if page == "Home":
        st.subheader("Stock-Specific Analysis")
        st.write("Analyze sentiment for specific stocks and correlate with price movements.")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            
            if not news_df.empty and 'ticker' in news_df.columns:
                available_tickers = sorted(news_df['ticker'].unique())
                ticker = st.selectbox(
                    "Select a stock ticker:",
                    options=available_tickers,
                    index=0 if available_tickers else None
                )
            else:
                ticker = st.text_input(
                    "Enter a stock ticker (e.g., AAPL, MSFT, GOOGL):",
                    value="AAPL"
                )
        
        with col2:
            start = st.date_input(
                "Start date:",
                datetime.now() - timedelta(days=30)
            )
        
        with col3:
            end = st.date_input(
                "End date:",
                datetime.now()
            )
        
        if st.button("Run Analysis", type="primary"):
            if ticker and not news_df.empty:
                with st.spinner(f"Fetching data for {ticker}..."):
                    stock_df = fetch_stock_data(ticker, start, end)
                    
                    if not stock_df.empty:
                        merged = correlate_sentiment_with_returns(news_df, stock_df, ticker)
                        display_analysis_results(merged, ticker)
                    else:
                        st.error(f"Could not fetch stock data for {ticker}")
                        st.info("Please check if the ticker symbol is correct and try a different date range.")
            else:
                st.warning("Please enter a valid ticker symbol.")
    
    elif page == "Custom Analysis":
        show_custom_analysis_page(vader, finbert)
    
    elif page == "Market Overview":
        show_market_overview_page(news_df)
    
if __name__ == "__main__":
    main()