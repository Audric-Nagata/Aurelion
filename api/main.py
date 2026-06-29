import logging
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI

from agents import AgentState, run_critic, run_quant_engineer, run_research, run_risk
from tools import generate_report

logger = logging.getLogger(__name__)

app = FastAPI(title="Aurelion", description="AI Quantitative Research Agent — autonomous strategy development and backtesting")

MAX_CRITIQUE_RETRIES = 3


@dataclass
class ResearchRequest:
    objective: str


@dataclass
class ResearchResponse:
    report: str
    file_path: str = ""


@dataclass
class HealthResponse:
    status: str


# ponytail: for loop over LangGraph StateGraph — 40 lines of graph plumbing for one conditional loop
def run_pipeline(state: AgentState) -> AgentState:
    state.update(run_research(state))
    for attempt in range(MAX_CRITIQUE_RETRIES + 1):
        state.update(run_quant_engineer(state))
        state.update(run_risk(state))
        state.update(run_critic(state))
        critique = state.get("critique", {})
        if critique.get("severity") != "critical":
            break
        state["critique_retries"] = attempt + 1
        logger.info("Critique critical (attempt %d/%d), revising", attempt + 1, MAX_CRITIQUE_RETRIES)
    state["final_report"] = generate_report(state)
    return state


INITIAL_STATE = AgentState(
    research_objective="",
    retrieved_papers=[],
    hypothesis="",
    strategy_code="",
    tickers=[],
    start_date="",
    end_date="",
    market_data={},
    features={},
    backtest_results={},
    risk_metrics={},
    critique={},
    final_report="",
    messages=[],
    critique_retries=0,
)


@app.get("/health")
def health():
    return HealthResponse(status="ok")


@app.post("/research")
def research(request: ResearchRequest):
    state = INITIAL_STATE.copy()
    state["research_objective"] = request.objective
    try:
        result = run_pipeline(state)
        report = result["final_report"]
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        filepath = output_dir / "research_report.md"
        filepath.write_text(report, encoding="utf-8")
        return ResearchResponse(report=report, file_path=str(filepath))
    except Exception as e:
        logger.exception("Research pipeline failed")
        return ResearchResponse(
            report=f"# Research Pipeline Error\n\n**Error type:** `{type(e).__name__}`\n\n**Message:** {e}\n"
        )
