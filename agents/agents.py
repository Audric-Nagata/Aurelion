import json
from typing import Any, TypedDict

import pandas as pd

from config import AGENT_MODELS
from models.llm_client import llm_chat
from tools import (
    compute_risk_metrics, evaluate_strategy, extract_code,
    fetch_market_data, generate_features, parse_json_block,
    run_backtest, search_research_papers,
)

# ponytail: functions over classes — BaseAgent, LLMClient, and 4 agent classes
#           replaced by 4 functions + 1 TypedDict. Same pipeline, half the files.


class AgentState(TypedDict):
    research_objective: str
    retrieved_papers: list[dict]
    hypothesis: str
    strategy_code: str
    tickers: list[str]
    start_date: str
    end_date: str
    market_data: dict
    features: dict
    backtest_results: dict
    risk_metrics: dict
    critique: dict
    final_report: str
    messages: list[dict]
    critique_retries: int


def run_research(state: AgentState) -> dict[str, Any]:
    objective = state.get("research_objective", "")
    papers = search_research_papers(objective)
    summaries = "\n".join(
        f"- {p.get('title', 'Untitled')}: {p.get('abstract', p.get('content', 'No content')[:200])}"
        for p in papers
    )
    messages = [
        {"role": "system", "content": "You are a quantitative research specialist. Search academic papers to find evidence-backed strategies and factors. Extract key hypotheses, signals, and methodologies. Output a structured hypothesis with supporting citations."},
        {"role": "user", "content": f"Research Objective: {objective}\n\nRelevant Papers:\n{summaries}\n\nForm a clear, testable trading hypothesis. Include:\n1. Hypothesis statement\n2. Specific signals/factors to test\n3. Which tickers and date range to use\n4. Suggested entry/exit logic"},
    ]
    hypothesis = llm_chat(messages, model=AGENT_MODELS["research"])
    return {"retrieved_papers": papers, "hypothesis": hypothesis, "messages": messages + [{"role": "assistant", "content": hypothesis}]}


def run_quant_engineer(state: AgentState) -> dict[str, Any]:
    state = dict(state)
    state.update(_fetch_data(state))
    state.update(_engineer_features(state))
    state.update(_write_strategy(state))
    state.update(_run_backtest(state))
    return state


def _fetch_data(state: AgentState) -> dict:
    messages = [
        {"role": "system", "content": 'Extract tickers, start date, and end date from the research objective and hypothesis. Respond in JSON: {"tickers": [...], "start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}'},
        {"role": "user", "content": f"Objective: {state.get('research_objective', '')}\nHypothesis: {state.get('hypothesis', '')}"},
    ]
    parsed = parse_json_block(llm_chat(messages, model=AGENT_MODELS["quant_engineer"], temperature=0.1))
    tickers = parsed.get("tickers", ["SPY"])
    start = parsed.get("start", "2020-01-01")
    end = parsed.get("end", "2025-01-01")
    return {"tickers": tickers, "start_date": start, "end_date": end, "market_data": fetch_market_data(tickers, start, end)}


def _engineer_features(state: AgentState) -> dict:
    return {"features": generate_features(state.get("market_data", {}))}


def _write_strategy(state: AgentState) -> dict:
    features = state.get("features", {})
    first = next(iter(features)) if features else ""
    df = pd.DataFrame(features[first]) if first else pd.DataFrame()
    columns = list(df.columns)
    sample = df.iloc[0].to_dict() if not df.empty else {}
    messages = [
        {"role": "system", "content": "You are a quantitative strategy developer. Write a Python function `generate_signals(df) -> pd.Series` that returns -1 (short), 0 (neutral), or 1 (long) signals. Use pandas and numpy."},
        {"role": "user", "content": f"Hypothesis: {state.get('hypothesis', '')}\n\nThe DataFrame `df` has these columns ONLY:\n  {columns}\n\nSample row: {sample}\n\nIMPORTANT:\n- Every column access must be `df['...']` with a name from the column list above.\n- Do NOT reference VIX, MACD, ATR, Bollinger Bands, or any column not listed.\n- Use `df['Close']`, not `df['AAPL_Close']` or `df['SPY_Close']`. No ticker prefixes.\n- Compute any derived values (e.g. MACD, Bollinger) from the available columns.\n- Write ONLY the Python code for `generate_signals`. No explanation, no markdown formatting."},
    ]
    code = extract_code(llm_chat(messages, model=AGENT_MODELS["quant_engineer"], temperature=0.3))
    # ponytail: syntax guard retries once instead of crashing in exec()
    try:
        compile(code, "<strategy>", "exec")
    except SyntaxError as e:
        messages += [
            {"role": "assistant", "content": code},
            {"role": "user", "content": f"Your code has a syntax error: {e}. Fix it and return ONLY valid Python code, no explanation."},
        ]
        code = extract_code(llm_chat(messages, model=AGENT_MODELS["quant_engineer"], temperature=0.3))
    return {"strategy_code": code}


def _run_backtest(state: AgentState) -> dict:
    return {"backtest_results": run_backtest(state.get("features", {}), state.get("strategy_code", ""))}


def run_risk(state: AgentState) -> dict[str, Any]:
    metrics = evaluate_strategy(state.get("backtest_results", {}))
    all_returns = []
    for ticker, records in state.get("market_data", {}).items():
        df = pd.DataFrame(records)
        if "Close" in df.columns and len(df) > 1:
            all_returns.extend(df["Close"].pct_change().dropna().tolist())
    if all_returns:
        metrics.update(compute_risk_metrics(all_returns))
    messages = [
        {"role": "system", "content": "You are a quantitative risk analyst. Evaluate strategy performance from a risk perspective. Flag any metrics that indicate unsustainable or dangerous behavior. Be conservative. A good Sharpe ratio does not justify high drawdown."},
        {"role": "user", "content": f"Backtest Results:\n{json.dumps(state.get('backtest_results', {}), indent=2)}\n\nRisk Metrics:\n{json.dumps(metrics, indent=2)}\n\nProvide a brief risk assessment. Keep it concise — 2-3 sentences."},
    ]
    metrics["interpretation"] = llm_chat(messages, model=AGENT_MODELS["risk"])
    return {"risk_metrics": metrics}


def run_critic(state: AgentState) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": "You are a skeptical quantitative reviewer. Your job is to find flaws, not validate. Check for data leakage, look-ahead bias, survivorship bias, and overfitting. Challenge every assumption. If the strategy looks too good, it probably is. Produce a structured critique with severity ratings: critical / warning / note."},
        {"role": "user", "content": f"Strategy Code:\n{state.get('strategy_code', '')}\n\nBacktest Results:\n{json.dumps(state.get('backtest_results', {}), indent=2)}\n\nRisk Metrics:\n{json.dumps(state.get('risk_metrics', {}), indent=2)}\n\nRespond in JSON format:\n{{\n  \"severity\": \"critical\" | \"warning\" | \"note\",\n  \"flags\": [\"list\", \"of\", \"issues\"],\n  \"recommendations\": \"brief suggestion\"\n}}"},
    ]
    critique = parse_json_block(llm_chat(messages, model=AGENT_MODELS["critic"], temperature=0.1))
    if "severity" not in critique:
        critique = {"severity": "note", "flags": ["Could not parse structured critique from LLM response"], "recommendations": "Manual review recommended."}
    return {"critique": critique}
