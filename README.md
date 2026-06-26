# Aurelion

**AI Quantitative Research Agent** — autonomous strategy development, backtesting, and reporting.

## Quick Start

```bash
cp .env.example .env
# Edit .env with your OpenRouter API key
pip install -r requirements.txt
uvicorn api.main:app --reload
```

## Usage

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"objective": "Research momentum strategies for AAPL, MSFT, GOOGL from 2020-2025"}'
```

## Architecture

```
FastAPI → QuantResearchAgent (LangGraph)
           ├── retrieve_papers (RAG)
           ├── form_hypothesis  (LLM)
           ├── fetch_data       (yfinance)
           ├── engineer_features
           ├── write_strategy   (LLM)
           ├── run_backtest     (VectorBT)
           ├── evaluate
           ├── critique         (LLM)
           └── generate_report
```

## Docker

```bash
docker compose up --build
```

## Project Structure

```
Aurelion/
├── agents/          # BaseAgent + QuantResearchAgent
├── tools/           # market_data, feature_engineering, backtesting, etc.
├── pipelines/       # Document ingestion + embedding
├── models/          # LLM client (OpenRouter)
├── api/             # FastAPI entry point
├── db/              # PostgreSQL + pgvector
├── config/          # Settings (pydantic-settings)
└── tests/
```
