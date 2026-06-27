import logging
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from agents.quant_research_agent import QuantResearchAgent
from config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an autonomous quantitative research agent.\n"
    "Your job is to research, build, and evaluate algorithmic "
    "trading strategies.\n"
    "Be rigorous, skeptical, and scientific."
)

app = FastAPI(
    title="Aurelion",
    description="AI Quantitative Research Agent — "
    "autonomous strategy development and backtesting",
)

agent = QuantResearchAgent(
    model=settings.default_llm_model,
    system_prompt=SYSTEM_PROMPT,
)


class ResearchRequest(BaseModel):
    objective: str


class ResearchResponse(BaseModel):
    report: str
    file_path: str = ""


class HealthResponse(BaseModel):
    status: str


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
    }
    try:
        result = agent.run(state)
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
