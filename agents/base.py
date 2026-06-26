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


class BaseAgent:
    def __init__(self, model: str, system_prompt: str):
        self.model = model
        self.system_prompt = system_prompt

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError
