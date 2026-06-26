from typing import Any


def evaluate_strategy(backtest_results: dict) -> dict[str, Any]:
    """Compute comprehensive risk and performance metrics."""
    metrics: dict[str, Any] = {}

    per_ticker = backtest_results.get("per_ticker", {})

    for ticker, result in per_ticker.items():
        metrics[f"{ticker}_sharpe"] = result["sharpe_ratio"]
        metrics[f"{ticker}_max_dd"] = result["max_drawdown"]
        metrics[f"{ticker}_win_rate"] = result["win_rate"]
        metrics[f"{ticker}_num_trades"] = result["num_trades"]

    metrics["avg_sharpe"] = backtest_results.get("avg_sharpe", 0)
    metrics["avg_max_drawdown"] = backtest_results.get("avg_max_drawdown", 0)
    metrics["total_return_pct"] = backtest_results.get("total_return_pct", 0)

    if metrics["avg_sharpe"] < 0:
        metrics["risk_flag"] = "Negative Sharpe — strategy loses to risk-free rate"
    elif metrics["avg_sharpe"] < 1:
        metrics["risk_flag"] = "Below-average risk-adjusted returns"
    else:
        metrics["risk_flag"] = "Acceptable risk-adjusted returns"

    if metrics["avg_max_drawdown"] < -30:
        metrics["drawdown_flag"] = "Critical — drawdown exceeds 30%"
    elif metrics["avg_max_drawdown"] < -15:
        metrics["drawdown_flag"] = "Warning — drawdown exceeds 15%"
    else:
        metrics["drawdown_flag"] = "Acceptable drawdown"

    max_dd = abs(metrics["avg_max_drawdown"])
    metrics["calmar_ratio"] = (
        metrics["total_return_pct"] / max_dd if max_dd > 0 else 0
    )

    return metrics
