import json
import re
from typing import Any

from langgraph.graph import END, StateGraph

from agents.base import AgentState, BaseAgent
from models.llm_client import LLMClient
from tools.backtesting import run_backtest
from tools.evaluation import evaluate_strategy
from tools.feature_engineering import generate_features
from tools.market_data import fetch_market_data
from tools.rag_search import search_research_papers
from tools.report_generator import generate_report


class QuantResearchAgent(BaseAgent):
    def __init__(self, model: str, system_prompt: str):
        super().__init__(model, system_prompt)
        self.llm = LLMClient(model=model)
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        builder = StateGraph(AgentState)

        builder.add_node("retrieve_papers", self._retrieve_papers)
        builder.add_node("form_hypothesis", self._form_hypothesis)
        builder.add_node("fetch_data", self._fetch_data)
        builder.add_node("engineer_features", self._engineer_features)
        builder.add_node("write_strategy", self._write_strategy)
        builder.add_node("run_backtest", self._run_backtest)
        builder.add_node("evaluate", self._evaluate)
        builder.add_node("critique", self._critique)
        builder.add_node("generate_report", self._generate_report)

        builder.set_entry_point("retrieve_papers")
        builder.add_edge("retrieve_papers", "form_hypothesis")
        builder.add_edge("form_hypothesis", "fetch_data")
        builder.add_edge("fetch_data", "engineer_features")
        builder.add_edge("engineer_features", "write_strategy")
        builder.add_edge("write_strategy", "run_backtest")
        builder.add_edge("run_backtest", "evaluate")
        builder.add_edge("evaluate", "critique")
        builder.add_edge("critique", "generate_report")
        builder.add_edge("generate_report", END)

        return builder.compile()

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        return self.graph.invoke(state)

    def _retrieve_papers(self, state: AgentState) -> dict:
        papers = search_research_papers(state["research_objective"])
        return {"retrieved_papers": papers}

    def _form_hypothesis(self, state: AgentState) -> dict:
        paper_summaries = "\n".join(
            f"- {p.get('title', 'Untitled')}: "
            f"{p.get('abstract', p.get('content', 'No content')[:200])}"
            for p in state.get("retrieved_papers", [])
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"Research Objective: {state['research_objective']}\n\n"
                    f"Relevant Papers:\n{paper_summaries}\n\n"
                    "Form a clear, testable trading hypothesis. Include:\n"
                    "1. Hypothesis statement\n"
                    "2. Specific signals/factors to test\n"
                    "3. Which tickers and date range to use\n"
                    "4. Suggested entry/exit logic"
                ),
            },
        ]

        hypothesis = self.llm.chat(messages)
        return {"hypothesis": hypothesis, "messages": messages + [
            {"role": "assistant", "content": hypothesis}
        ]}

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
                    f"Objective: {state['research_objective']}\n"
                    f"Hypothesis: {state.get('hypothesis', '')}"
                ),
            },
        ]

        response = self.llm.chat(messages, temperature=0.1)
        parsed = self._parse_json(response)

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
        features = generate_features(state["market_data"])
        return {"features": features}

    def _write_strategy(self, state: AgentState) -> dict:
        first_ticker = next(iter(state["features"]))
        sample_records = state["features"][first_ticker]
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
                    f"Hypothesis: {state['hypothesis']}\n\n"
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
        code = self._extract_code(code)

        return {"strategy_code": code}

    def _run_backtest(self, state: AgentState) -> dict:
        results = run_backtest(state["features"], state["strategy_code"])
        return {"backtest_results": results}

    def _evaluate(self, state: AgentState) -> dict:
        metrics = evaluate_strategy(state["backtest_results"])
        return {"risk_metrics": metrics}

    def _critique(self, state: AgentState) -> dict:
        backtest_json = json.dumps(state["backtest_results"], indent=2)
        risk_json = json.dumps(state["risk_metrics"], indent=2)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a skeptical quantitative reviewer. "
                    "Check for data leakage, look-ahead bias, "
                    "survivorship bias, overfitting, and unrealistic "
                    "assumptions. Provide a structured critique with "
                    "severity ratings: critical / warning / note."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Strategy Code:\n{state['strategy_code']}\n\n"
                    f"Backtest Results:\n{backtest_json}\n\n"
                    f"Risk Metrics:\n{risk_json}"
                ),
            },
        ]

        critique = self.llm.chat(messages)
        return {"critique": {"review": critique}}

    def _generate_report(self, state: AgentState) -> dict:
        report = generate_report(state)
        return {"final_report": report}

    @staticmethod
    def _extract_code(text: str) -> str:
        pattern = r"```python\n(.*?)\n```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)
        return text.strip()

    @staticmethod
    def _parse_json(text: str) -> dict:
        pattern = r"\{.*\}"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {}
