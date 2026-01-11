#Run this on a streamlit terminal first (streamlit run dashboard.py)

import streamlit as st
import os
import google.generativeai as genai
from google.oauth2 import service_account
import json
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    st.error("GOOGLE_API_KEY not found. Please set it in environment variables.")
else:
    genai.configure(api_key=api_key)

# Page configuration
st.set_page_config(
    page_title="AI Finance Assistant",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    background: linear-gradient(90deg, #1f77b4, #ff7f0e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 2rem;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
}
.sidebar .sidebar-content {
    background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
}
</style>
""", unsafe_allow_html=True)

## function to load Gemini Pro model and get responses
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

def get_stock_data_for_ai(ticker):
    """Fetch stock data for AI context"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="1mo")
        
        # Create a summary of key data
        current_price = info.get('currentPrice', 'N/A')
        market_cap = info.get('marketCap', 'N/A')
        pe_ratio = info.get('trailingPE', 'N/A')
        dividend_yield = info.get('dividendYield', 0)
        fifty_two_week_high = info.get('fiftyTwoWeekHigh', 'N/A')
        fifty_two_week_low = info.get('fiftyTwoWeekLow', 'N/A')
        
        # Recent price change
        if len(hist) > 1:
            price_change = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100
        else:
            price_change = 0
            
        summary = f"""
        Stock: {ticker}
        Company: {info.get('shortName', ticker)}
        Current Price: ${current_price}
        Market Cap: ${market_cap:,} if isinstance(market_cap, (int, float)) else market_cap
        P/E Ratio: {pe_ratio}
        52-Week High: ${fifty_two_week_high}
        52-Week Low: ${fifty_two_week_low}
        Dividend Yield: {dividend_yield*100:.2f}% if dividend_yield else 'N/A'
        Recent Price Change: {price_change:.2f}%
        Sector: {info.get('sector', 'N/A')}
        Industry: {info.get('industry', 'N/A')}
        """
        return summary
    except:
        return f"Could not fetch data for {ticker}"

def extract_ticker_from_question(question):
    """Extract ticker symbol from user question"""
    import re
    # Look for ticker patterns (2-5 uppercase letters)
    ticker_pattern = r'\b[A-Z]{2,5}\b'
    tickers = re.findall(ticker_pattern, question.upper())
    
    # Common stock tickers (you can expand this list)
    known_tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
                     'AMD', 'INTC', 'CRM', 'ORCL', 'IBM', 'DIS', 'NETFLIX']
    
    for ticker in tickers:
        if ticker in known_tickers or len(ticker) <= 5:
            return ticker
    return None

def get_gemini_response(question):
    # Check if question mentions a specific stock
    ticker = extract_ticker_from_question(question)
    
    # If ticker found, fetch current data
    if ticker:
        stock_data = get_stock_data_for_ai(ticker)
        enhanced_question = f"""
        Here is the current real-time data from Yahoo Finance for {ticker}:
        {stock_data}
        
        User question: {question}
        
        Please provide a comprehensive answer using this current data.
        """
    else:
        # Check if question is about finance and suggest they specify a ticker
        finance_keywords = ['stock', 'invest', 'buy', 'sell', 'price', 'market', 'finance', 'trading']
        if any(keyword in question.lower() for keyword in finance_keywords):
            enhanced_question = f"""
            {question}
            
            Note: If you're asking about a specific stock, please mention the ticker symbol (e.g., AAPL, GOOGL)
            and I can provide current real-time data from Yahoo Finance.
            """
        else:
            enhanced_question = question
    
    response = chat.send_message(enhanced_question, stream=True)
    return response

class FinanceAssistant:
    def __init__(self):
        self.stock_data = None
        self.stock_info = None

    def get_stock_data(self, ticker, period="1y"):
        try:
            stock = yf.Ticker(ticker)
            self.stock_data = stock.history(period=period)
            self.stock_info = stock.info
            return True
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {str(e)}")
            return False

    def create_stock_chart(self):
        if self.stock_data is None or self.stock_data.empty:
            return None

        fig = go.Figure()

        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=self.stock_data.index,
            open=self.stock_data['Open'],
            high=self.stock_data['High'],
            low=self.stock_data['Low'],
            close=self.stock_data['Close'],
            name="Price"
        ))

        # Volume chart
        fig.add_trace(go.Scatter(
            x=self.stock_data.index,
            y=self.stock_data['Volume'],
            mode='lines',
            name='Volume',
            yaxis='y2',
            line=dict(color='orange', width=1)
        ))

        fig.update_layout(
            title="Stock Price & Volume",
            yaxis_title="Price ($)",
            yaxis2=dict(
                title="Volume",
                overlaying='y',
                side='right'
            ),
            xaxis_title="Date",
            template="plotly_dark",
            height=500
        )

        return fig

    def get_financial_metrics(self):
        if not self.stock_info:
            return {}

        metrics = {}
        try:
            metrics['Current Price'] = f"${self.stock_info.get('currentPrice', 'N/A')}"
            metrics['Market Cap'] = f"${self.stock_info.get('marketCap', 0):,}"
            metrics['P/E Ratio'] = self.stock_info.get('trailingPE', 'N/A')
            metrics['52 Week High'] = f"${self.stock_info.get('fiftyTwoWeekHigh', 'N/A')}"
            metrics['52 Week Low'] = f"${self.stock_info.get('fiftyTwoWeekLow', 'N/A')}"
            metrics['Dividend Yield'] = f"{self.stock_info.get('dividendYield', 0) * 100:.2f}%" if self.stock_info.get('dividendYield') else 'N/A'
        except:
            pass

        return metrics

def calculate_investment_score(stock_info):
    score = 0
    reasons = []

    pe_ratio = stock_info.get('trailingPE')
    if pe_ratio:
        if pe_ratio < 15:
            score += 2
            reasons.append(f"✅ Low P/E ratio ({pe_ratio:.1f}) - undervalued")
        elif pe_ratio < 25:
            score += 1
            reasons.append(f"✅ Reasonable P/E ratio ({pe_ratio:.1f})")
        else:
            reasons.append(f"⚠️ High P/E ratio ({pe_ratio:.1f}) - may be overvalued")

    profit_margins = stock_info.get('profitMargins')
    if profit_margins:
        if profit_margins > 0.20:
            score += 2
            reasons.append(f"✅ Excellent profit margins ({profit_margins:.1%})")
        elif profit_margins > 0.10:
            score += 1
            reasons.append(f"✅ Good profit margins ({profit_margins:.1%})")
        else:
            reasons.append(f"⚠️ Low profit margins ({profit_margins:.1%})")

    roe = stock_info.get('returnOnEquity')
    if roe:
        if roe > 0.15:
            score += 1
            reasons.append(f"✅ Strong ROE ({roe:.1%})")
        elif roe > 0.10:
            reasons.append(f"✅ Decent ROE ({roe:.1%})")
        else:
            reasons.append(f"⚠️ Weak ROE ({roe:.1%})")

    debt_equity = stock_info.get('debtToEquity')
    if debt_equity:
        if debt_equity < 0.3:
            score += 1
            reasons.append(f"✅ Low debt-to-equity ({debt_equity:.1f})")
        elif debt_equity < 0.6:
            reasons.append(f"✅ Moderate debt-to-equity ({debt_equity:.1f})")
        else:
            reasons.append(f"⚠️ High debt-to-equity ({debt_equity:.1f})")

    dividend_yield = stock_info.get('dividendYield')
    if dividend_yield and dividend_yield > 0.02:
        score += 1
        reasons.append(f"✅ Pays dividend ({dividend_yield:.1%})")

    return score, reasons

def get_investment_recommendation(score, reasons):
    if score >= 5:
        return "🚀 STRONG BUY", "YES", reasons
    elif score >= 3:
        return "📈 BUY", "YES", reasons
    elif score >= 1:
        return "⚖️ HOLD/CAUTIOUS", "MAYBE", reasons
    else:
        return "❌ AVOID", "NO", reasons

# Initialize app
assistant = FinanceAssistant()



st.markdown('<h1 class="main-header">Finance Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📊 Stock Selection")
    popular_tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
    ticker = st.selectbox("Choose a stock ticker:", options=[''] + popular_tickers)
    custom_ticker = st.text_input("Or enter custom ticker:", placeholder="e.g., RELIANCE.NS")
    
    if custom_ticker:
        ticker = custom_ticker.upper()
    
    period = st.selectbox("Select time period:", options=['1mo', '2mo','3mo', '5mo','6mo', '1y', '2y', '5y'], index=3)
    
    if ticker and st.button("Get Stock Data", type="primary"):
        with st.spinner("Fetching stock data..."):
            if assistant.get_stock_data(ticker, period):
                st.success(f"Data loaded for {ticker}")
            else:
                st.error("Failed to load data")

# Main content
if assistant.stock_info:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.title(f"{assistant.stock_info.get('shortName', ticker)} ({ticker})")
        st.write(f"**Sector:** {assistant.stock_info.get('sector', 'N/A')} | **Industry:** {assistant.stock_info.get('industry', 'N/A')}")
    
    with col2:
        current_price = assistant.stock_info.get('currentPrice', 0)
        st.metric("Current Price", f"${current_price}", f"{assistant.stock_info.get('regularMarketChangePercent', 0):.2f}%")

    # Updated to 4 tabs including AI Assistant
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Technical Analysis", "🎯 Investment Insights", "💬 AI Assistant"])

    with tab1:
        st.subheader("📊 Key Financial Metrics")
        metrics = assistant.get_financial_metrics()
        cols = st.columns(3)
        metric_items = list(metrics.items())
        
        for i, (key, value) in enumerate(metric_items):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{key}</h4>
                    <h3>{value}</h3>
                </div>
                """, unsafe_allow_html=True)

        st.subheader("📈 Stock Performance")
        chart = assistant.create_stock_chart()
        if chart:
            st.plotly_chart(chart, use_container_width=True)

        if assistant.stock_info.get('longBusinessSummary'):
            st.subheader("🏢 Business Summary")
            st.write(assistant.stock_info['longBusinessSummary'])

    with tab2:
        st.header("📈 Technical Analysis")
        hist_data = assistant.stock_data.copy()
        
        if not hist_data.empty:
            hist_data['MA_20'] = hist_data['Close'].rolling(window=20).mean()
            hist_data['MA_50'] = hist_data['Close'].rolling(window=50).mean()

        # Trend insight logic
        trend_label = ""
        if hist_data['MA_20'].iloc[-1] > hist_data['MA_50'].iloc[-1]:
            trend_label = "📈 Bullish Trend (20-day MA above 50-day MA)"
        elif hist_data['MA_20'].iloc[-1] < hist_data['MA_50'].iloc[-1]:
            trend_label = "📉 Bearish Trend (20-day MA below 50-day MA)"
        else:
            trend_label = "⚖️ Neutral Trend"

        st.markdown(f"**Trend Insight:** {trend_label}")
        st.markdown("*Note: This trend is based on recent price behavior (moving averages). For a holistic view, consider the investment score based on fundamentals.*")

        # Chart plotting
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_data.index, y=hist_data['Close'], name='Close Price', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=hist_data.index, y=hist_data['MA_20'], name='20-day MA', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=hist_data.index, y=hist_data['MA_50'], name='50-day MA', line=dict(color='red')))

        fig.update_layout(
            title=f"{ticker} Technical Analysis",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template="plotly_dark",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("🎯 Detailed Investment Analysis")
        score, reasons = calculate_investment_score(assistant.stock_info)
        recommendation, decision, analysis = get_investment_recommendation(score, reasons)

        if decision == "YES":
            st.success(f"{recommendation} (Score: {score}/7)")
        elif decision == "MAYBE":
            st.warning(f"{recommendation} (Score: {score}/7)")
        else:
            st.error(f"{recommendation} (Score: {score}/7)")

        st.markdown("**Analysis Factors:**")
        for reason in reasons:
            if "✅" in reason:
                st.success(reason)
            elif "⚠️" in reason:
                st.warning(reason)
            else:
                st.info(reason)

        st.subheader("📊 Detailed Financial Metrics")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Valuation:**")
            st.write(f"• P/E Ratio: {assistant.stock_info.get('trailingPE', 'N/A')}")
            st.write(f"• Market Cap: ${assistant.stock_info.get('marketCap', 'N/A'):,}" if assistant.stock_info.get('marketCap') else "• Market Cap: N/A")

        with col2:
            st.markdown("**Profitability:**")
            st.write(f"• Profit Margins: {f'{assistant.stock_info.get('profitMargins', 0):.1%}' if assistant.stock_info.get('profitMargins') else 'N/A'}")
            st.write(f"• Operating Margins: {f'{assistant.stock_info.get('operatingMargins', 0):.1%}' if assistant.stock_info.get('operatingMargins') else 'N/A'}")
            st.write(f"• ROE: {f'{assistant.stock_info.get('returnOnEquity', 0):.1%}' if assistant.stock_info.get('returnOnEquity') else 'N/A'}")

    with tab4:
              
        st.markdown(
        """
        <iframe src="http://localhost:8502" width="100%" height="800" frameborder="0"></iframe>
        """,
        unsafe_allow_html=True
    )

else: 
    st.markdown("""
    <div style="text-align: center; padding: 3rem;">
        <h2>Welcome to Your Finance Dashboard!💰</h2>
        <p style="font-size: 1.2rem; color: #666;">
            Select a stock ticker from the sidebar to get started with comprehensive financial analysis.
        </p>
        <br>
        <p><strong>What you can do:</strong></p>
        <ul style="text-align: left; max-width: 600px; margin: 0 auto;">
            <li>📊 View real-time stock data and interactive charts</li>
            <li>💰 Get detailed financial metrics and analysis</li>
            <li>📈 Analyze technical indicators and moving averages</li>
            <li>🎯 Get investment recommendations</li>
            <li>🏢 Read comprehensive business summaries</li>
            <li>💬 Chat with AI assistant about finance</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p><strong>Disclaimer:</strong> This tool provides information for educational purposes only.
    This cannot replace a qualified financial advisor before making investment decisions.</p>  
</div>
""", unsafe_allow_html=True)
