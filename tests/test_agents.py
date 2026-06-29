from agents import AgentState


def test_agent_state_keys():
    required_keys = [
        "research_objective",
        "retrieved_papers",
        "hypothesis",
        "strategy_code",
        "tickers",
        "start_date",
        "end_date",
        "market_data",
        "features",
        "backtest_results",
        "risk_metrics",
        "critique",
        "final_report",
        "messages",
        "critique_retries",
    ]
    for key in required_keys:
        assert key in AgentState.__annotations__
