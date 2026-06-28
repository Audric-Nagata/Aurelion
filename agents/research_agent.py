from typing import Any

from agents.base import AgentState, BaseAgent
from models.llm_client import LLMClient
from tools.rag_search import search_research_papers


class ResearchAgent(BaseAgent):
    def __init__(self, model: str):
        system_prompt = (
            "You are a quantitative research specialist.\n"
            "Search academic papers to find evidence-backed strategies and factors.\n"
            "Extract key hypotheses, signals, and methodologies.\n"
            "Output a structured hypothesis with supporting citations."
        )
        super().__init__(model, tools=[search_research_papers], system_prompt=system_prompt)
        self.llm = LLMClient(model=model)

    def run(self, state: AgentState) -> dict[str, Any]:
        objective = state.get("research_objective", "")
        papers = search_research_papers(objective)

        paper_summaries = "\n".join(
            f"- {p.get('title', 'Untitled')}: "
            f"{p.get('abstract', p.get('content', 'No content')[:200])}"
            for p in papers
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"Research Objective: {objective}\n\n"
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
        return {
            "retrieved_papers": papers,
            "hypothesis": hypothesis,
            "messages": messages + [{"role": "assistant", "content": hypothesis}],
        }
