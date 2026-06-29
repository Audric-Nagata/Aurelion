import json
import logging
import re
from typing import Any

import numpy as np
import pandas as pd

# ponytail: one-file tools — 8 single-function modules for 8 functions is filing cabinet, not engineering

logger = logging.getLogger(__name__)


# ── data ──

def extract_code(text: str) -> str:
    pattern = r"```python\n(.*?)\n```"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else text.strip()


def parse_json_block(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {}


# ── market data ──

def _flatten_columns(df):
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df


def fetch_market_data(tickers: list[str], start: str, end: str) -> dict:
    import yfinance as yf
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)
    if len(tickers) == 1:
        ticker = tickers[0]
        df = _flatten_columns(data.reset_index())
        df["ticker"] = ticker
        return {ticker: df.to_dict(orient="records")}
    result = {}
    for ticker in tickers:
        if ticker in data.columns.get_level_values(1):
            df = _flatten_columns(data.xs(ticker, axis=1, level=1).reset_index())
            df["ticker"] = ticker
            result[ticker] = df.to_dict(orient="records")
    return result


# ── feature engineering ──

def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def generate_features(market_data: dict) -> dict:
    result = {}
    for ticker, records in market_data.items():
        df = pd.DataFrame(records)
        df.sort_values("Date", inplace=True)
        df["returns_1d"] = df["Close"].pct_change()
        df["returns_5d"] = df["Close"].pct_change(5)
        df["returns_20d"] = df["Close"].pct_change(20)
        df["SMA_10"] = df["Close"].rolling(10).mean()
        df["SMA_20"] = df["Close"].rolling(20).mean()
        df["SMA_50"] = df["Close"].rolling(50).mean()
        df["SMA_200"] = df["Close"].rolling(200).mean()
        df["volatility_20d"] = df["returns_1d"].rolling(20).std()
        df["volatility_50d"] = df["returns_1d"].rolling(50).std()
        df["rsi_14"] = _compute_rsi(df["Close"], 14)
        df["volume_sma_20"] = df["Volume"].rolling(20).mean()
        df["volume_ratio"] = df["Volume"] / df["volume_sma_20"]
        df["high_low_pct"] = (df["High"] - df["Low"]) / df["Close"]
        df["close_open_pct"] = (df["Close"] - df["Open"]) / df["Open"]
        result[ticker] = df.dropna().to_dict(orient="records")
    return result


# ── backtesting ──

def run_backtest(features: dict, strategy_code: str, init_cash: float = 100_000.0) -> dict:
    import vectorbt as vbt
    per_ticker = {}
    for ticker, records in features.items():
        df = pd.DataFrame(records)
        local_vars = {"pd": pd, "np": np, "vbt": vbt, "df": df}
        exec(strategy_code, {"pd": pd, "np": np, "vbt": vbt}, local_vars)
        fn = local_vars.get("generate_signals")
        if fn is None:
            continue
        price = df.set_index("Date")["Close"]
        try:
            signals = fn(df)
        except Exception as e:
            raise ValueError(f"Strategy code referenced column '{e}'. Available: {df.columns.tolist()}") from e
        pf = vbt.Portfolio.from_signals(price, signals == 1, signals == -1, init_cash=init_cash / len(features), freq="D")
        stats = pf.stats()
        per_ticker[ticker] = {
            "total_return": float(stats.get("Total Return [%]", 0)),
            "sharpe_ratio": float(stats.get("Sharpe Ratio", 0)),
            "max_drawdown": float(stats.get("Max Drawdown [%]", 0)),
            "win_rate": float(stats.get("Win Rate [%]", 0)),
            "num_trades": int(stats.get("Num Trades", 0)),
            "final_value": float(pf.final_value()),
        }
    returns = [r["total_return"] for r in per_ticker.values()]
    sharpes = [r["sharpe_ratio"] for r in per_ticker.values()]
    drawdowns = [r["max_drawdown"] for r in per_ticker.values()]
    return {
        "per_ticker": per_ticker,
        "avg_return": sum(returns) / len(returns) if returns else 0,
        "avg_sharpe": sum(sharpes) / len(sharpes) if sharpes else 0,
        "avg_max_drawdown": sum(drawdowns) / len(drawdowns) if drawdowns else 0,
        "total_final_value": sum(r["final_value"] for r in per_ticker.values()),
        "total_return_pct": ((sum(r["final_value"] for r in per_ticker.values()) / init_cash) - 1) * 100,
    }


# ── evaluation ──

def evaluate_strategy(backtest_results: dict) -> dict[str, Any]:
    avg_sharpe = backtest_results.get("avg_sharpe", 0)
    avg_max_drawdown = backtest_results.get("avg_max_drawdown", 0)
    total_return_pct = backtest_results.get("total_return_pct", 0)
    max_dd = abs(avg_max_drawdown)
    return {
        "risk_flag": "Negative Sharpe — strategy loses to risk-free rate" if avg_sharpe < 0 else "Below-average risk-adjusted returns" if avg_sharpe < 1 else "Acceptable risk-adjusted returns",
        "drawdown_flag": "Critical — drawdown exceeds 30%" if avg_max_drawdown < -30 else "Warning — drawdown exceeds 15%" if avg_max_drawdown < -15 else "Acceptable drawdown",
        "calmar_ratio": total_return_pct / max_dd if max_dd > 0 else 0,
    }


# ── risk metrics ──

def compute_risk_metrics(returns: list[float]) -> dict:
    arr = np.array(returns)
    mu = np.mean(arr)
    sigma = np.std(arr, ddof=1)
    if sigma == 0 or len(arr) < 2:
        return {"var_95": 0, "var_99": 0, "cvar_95": 0, "sortino": 0, "annualized_vol": 0}
    var_95 = np.percentile(arr, 5)
    var_99 = np.percentile(arr, 1)
    cvar_95 = arr[arr <= var_95].mean() if np.any(arr <= var_95) else var_95
    downside = arr[arr < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1.0
    return {
        "var_95": round(float(var_95), 4),
        "var_99": round(float(var_99), 4),
        "cvar_95": round(float(cvar_95), 4),
        "sortino": round(float(sortino := mu / downside_std * np.sqrt(252)), 4),
        "annualized_vol": round(float(sigma * np.sqrt(252)), 4),
    }


# ── RAG search ──

def search_research_papers(query: str, top_k: int = 5) -> list[dict]:
    try:
        from models.embedding_client import EmbeddingClient
        from db.vector_store import VectorStore
        embedder = EmbeddingClient()
        vector_store = VectorStore()
        embedding = embedder.embed(query)
        return vector_store.search(embedding, top_k=top_k)
    except Exception as e:
        logger.warning("RAG search failed (pipeline will continue without papers): %s", e)
        return []


# ── report generator ──

def generate_report(state: dict[str, Any]) -> str:
    lines = [
        "# Quantitative Research Report",
        f"\n**Objective:** {state.get('research_objective', 'N/A')}",
        f"\n## Hypothesis\n{state.get('hypothesis', 'N/A')}",
        "\n## Strategy Code\n```python",
        state.get("strategy_code", "# No strategy generated"),
        "```",
        "\n## Backtest Results\n",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    bt = state.get("backtest_results", {})
    lines.append(f"| Avg Return | {bt.get('avg_return', 0):.2f}% |")
    lines.append(f"| Avg Sharpe | {bt.get('avg_sharpe', 0):.2f} |")
    lines.append(f"| Avg Max Drawdown | {bt.get('avg_max_drawdown', 0):.2f}% |")
    lines.append(f"| Total Return | {bt.get('total_return_pct', 0):.2f}% |")
    risk = state.get("risk_metrics", {})
    lines.append("\n## Risk Assessment\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Calmar Ratio | {risk.get('calmar_ratio', 0):.2f} |")
    lines.append(f"| Risk Flag | {risk.get('risk_flag', 'N/A')} |")
    lines.append(f"| Drawdown Flag | {risk.get('drawdown_flag', 'N/A')} |")
    critique = state.get("critique", {})
    lines.append("\n## Critique")
    if not critique:
        lines.append("No critique provided.")
    elif "severity" in critique:
        lines.append(f"**Severity:** {critique.get('severity', 'unknown')}")
        for flag in critique.get("flags", []):
            lines.append(f"- [x] {flag}")
        rec = critique.get("recommendations", "")
        if rec:
            lines.append(f"\n**Recommendation:** {rec}")
    else:
        for key, value in critique.items():
            lines.append(f"- **{key}:** {value}")
    lines.append("\n---")
    lines.append("*Generated by Aurelion Quant Research Agent*")
    return "\n".join(lines)
