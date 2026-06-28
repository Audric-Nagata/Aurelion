from agents.base import AgentState, BaseAgent


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


def test_base_agent_interface():
    class TestAgent(BaseAgent):
        def run(self, state):
            return state

    agent = TestAgent(model="test-model", tools=[], system_prompt="test")
    assert agent.model == "test-model"
    assert agent.system_prompt == "test"
    assert agent.run({"key": "value"}) == {"key": "value"}
