#Keep this running on another terminal (streamlit run chatbot.py)

import streamlit as st
import os
import yfinance as yf
import re
from datetime import datetime, timezone
from google import genai
from google.genai import types

st.set_page_config(page_title="Finance Chatbot", layout="centered")
st.markdown("<style>#MainMenu{visibility:hidden;}footer{visibility:hidden;}</style>", unsafe_allow_html=True)

api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEY not set")
    st.stop()

client = genai.Client(api_key=api_key)

def init_chat():
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = client.chats.create(
            model="gemini-2.0-flash-001",
            config=types.GenerateContentConfig(system_instruction="You are an expert finance assistant. Only respond to finance-related queries. You can make analysis and comparisons of the stocks of different companies for the given time period. Also give definitions to the financial terms when prompted.")
        )
    if "messages" not in st.session_state:
        st.session_state.messages = []

def show_history():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar=msg.get("avatar", "🤖")):
            st.markdown(msg["content"])

def welcome_once():
    if not st.session_state.messages:
        msg = "👋 Hello, Welcome! I'm your finance assistant.\n\nAsk about stock prices, comparisons, or financial metrics like P/E ratio or ROE."
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(msg)
        st.session_state.messages.append({"role": "assistant", "content": msg, "avatar": "🤖"})

def get_current_datetime():
    """Get current date and time information"""
    now = datetime.now()
    date_str = now.strftime("%A, %B %d, %Y")
    time_str = now.strftime("%I:%M %p")
    return f"📅 **Today's Date**: {date_str}\n🕐 **Current Time**: {time_str}"

def is_date_query(prompt):
    """Check if the query is asking for date/time information"""
    date_keywords = [
        "today", "date", "time", "current date", "what day", 
        "today's date", "current time", "what time"
    ]
    return any(keyword in prompt.lower() for keyword in date_keywords)

def extract_months(prompt):
    match = re.search(r"(\d+)\s*(month|months|mo)", prompt.lower())
    return int(match.group(1)) if match else 1

def extract_ticker(prompt):
    known = {
        "apple": "AAPL", "tesla": "TSLA", "microsoft": "MSFT",
        "google": "GOOG", "meta": "META", "amazon": "AMZN",
        "reliance": "RELIANCE.NS", "infosys": "INFY.NS",
        "tcs": "TCS.NS", "icici": "ICICIBANK.NS", "hdfc": "HDFCBANK.NS"
    }
    for word in prompt.lower().split():
        if word.upper() in known.values(): return word.upper()
        if word in known: return known[word]
    return None

def get_current_price(ticker):
    try:
        price = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
        return f"📈\nThe current stock price of **{ticker}** is **${price:.2f}**."
    except:
        return f"⚠️ Couldn't fetch current price for {ticker}."

def get_specific_metric(prompt, ticker):
    info = yf.Ticker(ticker).info
    metrics = {
        "pe ratio": "trailingPE",
        "p/e ratio": "trailingPE",
        "roe": "returnOnEquity",
        "dividend": "dividendYield",
        "eps": "trailingEps",
        "market cap": "marketCap",
        "volume": "averageVolume",
        "yield": "dividendYield",
    }
    for key, field in metrics.items():
        if key in prompt.lower():
            val = info.get(field)
            if val is None:
                return f"⚠️ {key.title()} not available for {ticker}."
            if 'yield' in key:
                val = f"{val*100:.2f}%"
            elif isinstance(val, float):
                val = f"{val:.2f}"
            return f"🌟\n**{key.title()} of {ticker}**: {val}"
    return None

def analyze_stock(prompt):
    ticker = extract_ticker(prompt)
    if not ticker: return None
    months = extract_months(prompt)
    period = "1mo" if months <= 1 else "3mo" if months <= 3 else "6mo" if months <= 6 else "1y"
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    info = stock.info
    if hist.empty: return f"⚠️ No data for {ticker}."
    start = hist['Close'].iloc[0]
    end = hist['Close'].iloc[-1]
    change = end - start
    pct = (change / start) * 100
    high = hist['High'].max()
    low = hist['Low'].min()
    volume = int(hist['Volume'].mean())
    trend = "⚠️ Significant Decline" if pct < -5 else "✅ Moderate Gain" if pct > 5 else "⏳ Stable"
    return f"""
📈\n📊 **{info.get('shortName','Stock')} ({ticker}) - Stock Analysis**

**Price Performance (Last {months} Month{'s' if months > 1 else ''})**
- Current Price: ${end:.2f}
- Starting Price: ${start:.2f}
- Price Change: ${change:+.2f} ({pct:+.2f}%)
- 52-Week High: ${info.get('fiftyTwoWeekHigh','N/A')}
- 52-Week Low: ${info.get('fiftyTwoWeekLow','N/A')}
- Average Daily Volume: {volume:,} shares

**Key Metrics**
- Market Cap: ${round(info.get('marketCap', 0)/1e12, 2)}T
- P/E Ratio: {info.get('trailingPE','N/A')}
- EPS: {info.get('trailingEps','N/A')}
- Dividend Yield: {round(info.get('dividendYield', 0)*100, 2)}%

**Recent Performance**
{trend} - Stock has {"risen" if pct > 0 else "dropped" if pct < 0 else "remained flat"} {pct:+.1f}% over the analyzed period.
"""

def compare_stocks(prompt):
    words = prompt.lower().split()
    tickers = list(set([extract_ticker(w) for w in words if extract_ticker(w)]))
    if len(tickers) < 2: return None
    t1, t2 = tickers[:2]
    months = extract_months(prompt)
    period = "1mo" if months <= 1 else "3mo" if months <= 3 else "6mo" if months <= 6 else "1y"
    s1, s2 = yf.Ticker(t1), yf.Ticker(t2)
    h1, h2 = s1.history(period=period), s2.history(period=period)
    i1, i2 = s1.info, s2.info
    try:
        p1s, p1e = h1['Close'].iloc[0], h1['Close'].iloc[-1]
        p2s, p2e = h2['Close'].iloc[0], h2['Close'].iloc[-1]
        ch1, ch2 = ((p1e - p1s)/p1s)*100, ((p2e - p2s)/p2s)*100
        best = t1 if ch1 > ch2 else t2
        vol1, vol2 = h1['High'].max() - h1['Low'].min(), h2['High'].max() - h2['Low'].min()
        volatile = t1 if vol1 > vol2 else t2
        stable = t2 if volatile == t1 else t1
        return f"""
📈\n📊 **Stock Comparison Report (Last {months} Months)**

📈 **Price Performance Comparison**
1. {t1} {'🔼' if ch1 > 0 else '🔽'}
Current Price: ${p1e:.2f}
Change: ${p1e - p1s:+.2f} ({ch1:+.2f}%)
High: ${h1['High'].max():.2f} ∣ Low: ${h1['Low'].min():.2f}

2. {t2} {'🔼' if ch2 > 0 else '🔽'}
Current Price: ${p2e:.2f}
Change: ${p2e - p2s:+.2f} ({ch2:+.2f}%)
High: ${h2['High'].max():.2f} ∣ Low: ${h2['Low'].min():.2f}

🏆 **Performance Summary**
Best Performer: {best} with {max(ch1,ch2):+.2f}% return
Worst Performer: {t1 if best == t2 else t2} with {min(ch1,ch2):+.2f}% return

📊 **Key Metrics Comparison**
**{t1}**
- Market Cap: ${round(i1.get('marketCap',0)/1e12,2)}T
- P/E Ratio: {i1.get('trailingPE','N/A')}
- Avg Volume: {int(h1['Volume'].mean()):,}

**{t2}**
- Market Cap: ${round(i2.get('marketCap',0)/1e12,2)}T
- P/E Ratio: {i2.get('trailingPE','N/A')}
- Avg Volume: {int(h2['Volume'].mean()):,}

💡 **Investment Insights**
Most Volatile: {volatile} (wider price range)
Most Stable: {stable} (narrower price range)
"""
    except Exception as e:
        return f"⚠️ Error comparing: {e}"

init_chat()
show_history()
welcome_once()

prompt = st.chat_input("Ask about stocks or finance terms...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = None
    
    if prompt.lower().strip() in ["hi", "hello", "hey"]:
        response = "👋 Hi! I'm your finance assistant. Ask me anything about stocks or finance terms."
    elif is_date_query(prompt):
        response = get_current_datetime()
    elif any(word in prompt.lower() for word in ["compare", "comparison", " vs ", " versus "]):
        response = compare_stocks(prompt)
    elif ticker := extract_ticker(prompt):
        if any(word in prompt.lower() for word in ["price", "current", "today", "now"]):
            response = get_current_price(ticker)
        else:
            response = get_specific_metric(prompt, ticker)
            if not response:
                response = analyze_stock(prompt)
    else:
        try:
            response = st.session_state.chat_session.send_message(prompt).text
        except Exception as e:
            response = f"⚠️ Gemini error: {e}"

    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response, "avatar": "🤖"})

st.divider()

if st.button("🔄 Start New Session"):
    st.session_state.pop("chat_session", None)
    st.session_state.pop("messages", None)
    st.rerun()