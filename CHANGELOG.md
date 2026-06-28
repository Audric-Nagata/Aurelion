# Changelog

## v0.1.0 — Phase 1 MVP (Single Agent)

Initial release of Aurelion — an autonomous quantitative research agent.

### Features

- **End-to-end quant research pipeline** — single agent runs the full workflow: paper retrieval, hypothesis generation, market data fetching, feature engineering, strategy writing, backtesting, evaluation, critique, and report generation
- **RAG-based paper search** — retrieves relevant academic papers from a pgvector database using sentence embeddings
- **Market data fetching** — pulls OHLCV data from yfinance for any ticker and date range
- **Feature engineering** — computes technical indicators (SMAs, RSI, volatility, volume ratios, returns)
- **Strategy backtesting** — executes LLM-generated strategy code via VectorBT; supports long/short signals
- **Performance evaluation** — computes Sharpe ratio, max drawdown, win rate, Calmar ratio, VaR, CVaR, Sortino ratio
- **Automated critique** — LLM-powered review checks for data leakage, look-ahead bias, survivorship bias, and overfitting
- **Structured Markdown reports** — generates a complete research report with hypothesis, code, backtest results, risk metrics, and critique
- **FastAPI web API** — `/health` and `/research` endpoints for triggering the pipeline
- **LangGraph orchestration** — state-machine-based workflow with 9 sequential nodes
- **Docker support** — `docker-compose.yml` with API + PostgreSQL/pgvector services
- **Model-agnostic LLM layer** — all models injectable via `.env`; no hardcoded model names

### Architecture

```
User / API Client
    |
FastAPI endpoint
    |
QuantResearchAgent (LangGraph)
    |
    ├── RAG search (pgvector)
    ├── Market data (yfinance)
    ├── Feature engineering (pandas/numpy)
    ├── Strategy backtest (VectorBT)
    ├── Risk evaluation
    ├── LLM critique
    └── Markdown report
```
