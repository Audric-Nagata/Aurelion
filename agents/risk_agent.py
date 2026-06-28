import json
from typing import Any

import pandas as pd

from agents.base import AgentState, BaseAgent
from models.llm_client import LLMClient
from tools.evaluation import evaluate_strategy
from tools.risk_metrics import compute_risk_metrics


class RiskAgent(BaseAgent):
    def __init__(self, model: str):
        system_prompt = (
            "You are a quantitative risk analyst.\n"
            "Evaluate strategy performance from a risk perspective.\n"
            "Flag any metrics that indicate unsustainable or dangerous behavior.\n"
            "Be conservative. A good Sharpe ratio does not justify high drawdown."
        )
        super().__init__(
            model,
            tools=[evaluate_strategy, compute_risk_metrics],
            system_prompt=system_prompt,
        )
        self.llm = LLMClient(model=model)

    def run(self, state: AgentState) -> dict[str, Any]:
        metrics = evaluate_strategy(state.get("backtest_results", {}))

        all_returns = []
        for ticker, records in state.get("market_data", {}).items():
            df = pd.DataFrame(records)
            if "Close" in df.columns and len(df) > 1:
                returns = df["Close"].pct_change().dropna().tolist()
                all_returns.extend(returns)

        if all_returns:
            extra = compute_risk_metrics(all_returns)
            metrics.update(extra)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"Backtest Results:\n"
                    f"{json.dumps(state.get('backtest_results', {}), indent=2)}\n\n"
                    f"Risk Metrics:\n{json.dumps(metrics, indent=2)}\n\n"
                    "Provide a brief risk assessment. "
                    "Keep it concise — 2-3 sentences."
                ),
            },
        ]

        interpretation = self.llm.chat(messages)
        metrics["interpretation"] = interpretation

        return {"risk_metrics": metrics}
