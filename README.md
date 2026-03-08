# AI Finance Assistant

An AI-powered financial analysis application that lets you query real-time stock data through natural language, analyze company fundamentals, and explore investment insights — all through an interactive dashboard and conversational chatbot.

---

## What it does

**Chatbot** — Ask questions like "Compare Apple and Microsoft" or "What's the PE ratio of Tesla?" and get structured, intelligent responses powered by Google Gemini AI. The chatbot handles intent detection so it knows whether you're asking for a price, a metric, or a full comparison.

**Dashboard** — Visual exploration of stock performance, market volatility, and investment metrics built with Plotly and Streamlit. Covers price trends, technical indicators, and fundamental data side by side.

**Real-time data** — All stock data is fetched live via yFinance, so prices and metrics are always current.

---

## Setup

```bash
git clone https://github.com/sreelakshmisajith05/finance_assistant
cd finance_assistant

pip install -r requirements.txt
```

Create a `.env` file in the root folder:

```
GOOGLE_API_KEY=your_api_key_here
```

Get your free API key at [Google AI Studio](https://aistudio.google.com/).

---

## Run

```bash
# Chatbot
streamlit run chatbot.py

# Dashboard
streamlit run dashboard.py
```

---

## Features

- Natural language stock queries — prices, PE ratio, ROE, dividend yield, and more
- Multi-stock comparisons with structured side-by-side reports
- Interactive charts for performance trends and market volatility
- Investment insights and analysis summaries
- Clean, responsive Streamlit UI

---

## Tech Stack

Python · Streamlit · Google Gemini AI · yFinance · Plotly

---

*For educational purposes only. Not financial advice.*
