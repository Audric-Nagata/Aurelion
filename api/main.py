import logging
from pathlib import Path

from fastapi import FastAPI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from agents import AgentState, CriticAgent, QuantEngineerAgent, ResearchAgent, RiskAgent
from config import AGENT_MODELS
from tools.report_generator import generate_report

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Aurelion",
    description="AI Quantitative Research Agent — "
    "autonomous strategy development and backtesting",
)

research_agent = ResearchAgent(model=AGENT_MODELS["research"])
quant_engineer_agent = QuantEngineerAgent(model=AGENT_MODELS["quant_engineer"])
risk_agent = RiskAgent(model=AGENT_MODELS["risk"])
critic_agent = CriticAgent(model=AGENT_MODELS["critic"])

MAX_CRITIQUE_RETRIES = 3


class ResearchRequest(BaseModel):
    objective: str


class ResearchResponse(BaseModel):
    report: str
    file_path: str = ""


class HealthResponse(BaseModel):
    status: str


def _decide_next(state: AgentState) -> str:
    critique = state.get("critique", {})
    severity = critique.get("severity", "note")
    retries = state.get("critique_retries", 0)

    if severity == "critical" and retries < MAX_CRITIQUE_RETRIES:
        logger.info("Critique critical (attempt %d/%d), revising", retries + 1, MAX_CRITIQUE_RETRIES)
        return "revise"
    logger.info("Critique %s, generating report", severity)
    return "report"


def _build_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("research", lambda state: research_agent.run(state))
    builder.add_node("quant_engineer", lambda state: quant_engineer_agent.run(state))
    builder.add_node("risk", lambda state: risk_agent.run(state))
    builder.add_node("critic", lambda state: critic_agent.run(state))
    builder.add_node("bump_retry", lambda state: {"critique_retries": state.get("critique_retries", 0) + 1})
    builder.add_node("generate_report", lambda state: {"final_report": generate_report(state)})

    builder.set_entry_point("research")
    builder.add_edge("research", "quant_engineer")
    builder.add_edge("quant_engineer", "risk")
    builder.add_edge("risk", "critic")
    builder.add_conditional_edges(
        "critic",
        _decide_next,
        {
            "revise": "bump_retry",
            "report": "generate_report",
        },
    )
    builder.add_edge("bump_retry", "quant_engineer")
    builder.add_edge("generate_report", END)

    return builder.compile()


graph = _build_graph()


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.post("/research", response_model=ResearchResponse)
def run_research(request: ResearchRequest):
    state = {
        "research_objective": request.objective,
        "retrieved_papers": [],
        "hypothesis": "",
        "strategy_code": "",
        "tickers": [],
        "start_date": "",
        "end_date": "",
        "market_data": {},
        "features": {},
        "backtest_results": {},
        "risk_metrics": {},
        "critique": {},
        "final_report": "",
        "messages": [],
        "critique_retries": 0,
    }
    try:
        result = graph.invoke(state)
        report = result["final_report"]

        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        filepath = output_dir / "research_report.md"
        filepath.write_text(report, encoding="utf-8")

        return ResearchResponse(report=report, file_path=str(filepath))
    except Exception as e:
        logger.exception("Research pipeline failed")
        return ResearchResponse(
            report=f"# Research Pipeline Error\n\n"
                   f"**Error type:** `{type(e).__name__}`\n\n"
                   f"**Message:** {e}\n"
        )
