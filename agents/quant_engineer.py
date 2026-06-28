from typing import Any

from agents.base import AgentState, BaseAgent
from models.llm_client import LLMClient
from tools.backtesting import run_backtest
from tools.feature_engineering import generate_features
from tools.market_data import fetch_market_data
from tools.utils import extract_code, parse_json_block


class QuantEngineerAgent(BaseAgent):
    def __init__(self, model: str):
        system_prompt = (
            "You are a quantitative engineer.\n"
            "Convert research hypotheses into production-grade trading strategies.\n"
            "Write clean, testable strategy code.\n"
            "Run backtests and report raw results without cherry-picking.\n"
            "Never optimize solely for in-sample performance."
        )
        super().__init__(
            model,
            tools=[fetch_market_data, generate_features, run_backtest],
            system_prompt=system_prompt,
        )
        self.llm = LLMClient(model=model)

    def run(self, state: AgentState) -> dict[str, Any]:
        state.update(self._fetch_data(state))
        state.update(self._engineer_features(state))
        state.update(self._write_strategy(state))
        state.update(self._run_backtest(state))
        return state

    def _fetch_data(self, state: AgentState) -> dict:
        messages = [
            {
                "role": "system",
                "content": (
                    "Extract tickers, start date, and end date from the "
                    "research objective and hypothesis. "
                    'Respond in JSON: {"tickers": [...], '
                    '"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Objective: {state.get('research_objective', '')}\n"
                    f"Hypothesis: {state.get('hypothesis', '')}"
                ),
            },
        ]

        response = self.llm.chat(messages, temperature=0.1)
        parsed = parse_json_block(response)

        tickers = parsed.get("tickers", ["SPY"])
        start = parsed.get("start", "2020-01-01")
        end = parsed.get("end", "2025-01-01")

        data = fetch_market_data(tickers, start, end)
        return {
            "tickers": tickers,
            "start_date": start,
            "end_date": end,
            "market_data": data,
        }

    def _engineer_features(self, state: AgentState) -> dict:
        features = generate_features(state.get("market_data", {}))
        return {"features": features}

    def _write_strategy(self, state: AgentState) -> dict:
        features = state.get("features", {})
        first_ticker = next(iter(features)) if features else ""
        sample_records = features[first_ticker] if first_ticker else []
        sample_df = __import__("pandas").DataFrame(sample_records)
        columns = list(sample_df.columns)
        sample_row = (
            sample_df.iloc[0].to_dict() if not sample_df.empty else {}
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a quantitative strategy developer. "
                    "Write a Python function `generate_signals(df) -> pd.Series` "
                    "that returns -1 (short), 0 (neutral), or 1 (long) signals. "
                    "Use pandas and numpy."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Hypothesis: {state.get('hypothesis', '')}\n\n"
                    "The DataFrame `df` has these columns ONLY:\n"
                    f"  {columns}\n\n"
                    f"Sample row: {sample_row}\n\n"
                    "IMPORTANT: Use column names EXACTLY as listed above. "
                    "For example, use `df['Close']`, not `df['AAPL_Close']` "
                    "or `df['SPY_Close']`. No ticker prefixes.\n\n"
                    "Write ONLY the Python code for `generate_signals`. "
                    "No explanation, no markdown formatting."
                ),
            },
        ]

        code = self.llm.chat(messages, temperature=0.3)
        code = extract_code(code)
        return {"strategy_code": code}

    def _run_backtest(self, state: AgentState) -> dict:
        results = run_backtest(
            state.get("features", {}),
            state.get("strategy_code", ""),
        )
        return {"backtest_results": results}

