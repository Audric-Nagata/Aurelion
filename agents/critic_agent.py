import json
from typing import Any

from agents.base import AgentState, BaseAgent
from models.llm_client import LLMClient
from tools.utils import parse_json_block


class CriticAgent(BaseAgent):
    def __init__(self, model: str):
        system_prompt = (
            "You are a skeptical quantitative reviewer.\n"
            "Your job is to find flaws, not validate.\n"
            "Check for data leakage, look-ahead bias, survivorship bias, and overfitting.\n"
            "Challenge every assumption. If the strategy looks too good, it probably is.\n"
            "Produce a structured critique with severity ratings: critical / warning / note."
        )
        super().__init__(model, tools=[], system_prompt=system_prompt)
        self.llm = LLMClient(model=model)

    def run(self, state: AgentState) -> dict[str, Any]:
        backtest_json = json.dumps(state.get("backtest_results", {}), indent=2)
        risk_json = json.dumps(state.get("risk_metrics", {}), indent=2)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"Strategy Code:\n{state.get('strategy_code', '')}\n\n"
                    f"Backtest Results:\n{backtest_json}\n\n"
                    f"Risk Metrics:\n{risk_json}\n\n"
                    'Respond in JSON format:\n'
                    '{\n'
                    '  "severity": "critical" | "warning" | "note",\n'
                    '  "flags": ["list", "of", "issues"],\n'
                    '  "recommendations": "brief suggestion"\n'
                    '}'
                ),
            },
        ]

        response = self.llm.chat(messages, temperature=0.1)
        critique = parse_json_block(response)

        if "severity" not in critique:
            critique = {
                "severity": "note",
                "flags": ["Could not parse structured critique from LLM response"],
                "recommendations": "Manual review recommended.",
            }

        return {"critique": critique}


