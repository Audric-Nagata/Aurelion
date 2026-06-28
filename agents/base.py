from collections.abc import Callable
from typing import Any, TypedDict


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


class BaseAgent:
    def __init__(self, model: str, tools: list[Callable], system_prompt: str):
        self.model = model
        self.tools = tools
        self.system_prompt = system_prompt

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError
