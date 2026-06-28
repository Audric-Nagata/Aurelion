from typing import Any


def evaluate_strategy(backtest_results: dict) -> dict[str, Any]:
    """Compute derived risk flags and Calmar ratio from backtest results.

    Raw per-ticker metrics (sharpe, drawdown, etc.) live in
    backtest_results['per_ticker'] — this function only computes
    synthesized flags and derived ratios.
    """
    avg_sharpe = backtest_results.get("avg_sharpe", 0)
    avg_max_drawdown = backtest_results.get("avg_max_drawdown", 0)
    total_return_pct = backtest_results.get("total_return_pct", 0)

    if avg_sharpe < 0:
        risk_flag = "Negative Sharpe — strategy loses to risk-free rate"
    elif avg_sharpe < 1:
        risk_flag = "Below-average risk-adjusted returns"
    else:
        risk_flag = "Acceptable risk-adjusted returns"

    if avg_max_drawdown < -30:
        drawdown_flag = "Critical — drawdown exceeds 30%"
    elif avg_max_drawdown < -15:
        drawdown_flag = "Warning — drawdown exceeds 15%"
    else:
        drawdown_flag = "Acceptable drawdown"

    max_dd = abs(avg_max_drawdown)
    calmar_ratio = total_return_pct / max_dd if max_dd > 0 else 0

    return {
        "risk_flag": risk_flag,
        "drawdown_flag": drawdown_flag,
        "calmar_ratio": calmar_ratio,
    }
