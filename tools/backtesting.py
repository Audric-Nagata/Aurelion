import pandas as pd
import numpy as np
import vectorbt as vbt


def run_backtest(
    features: dict,
    strategy_code: str,
    init_cash: float = 100_000.0,
) -> dict:
    """Execute a VectorBT backtest per ticker using strategy code."""
    per_ticker_results = {}

    for ticker, records in features.items():
        df = pd.DataFrame(records)

        local_vars = {"pd": pd, "np": np, "vbt": vbt, "df": df}
        exec(strategy_code, {"pd": pd, "np": np, "vbt": vbt}, local_vars)
        signals_fn = local_vars.get("generate_signals")

        if signals_fn is None:
            continue

        available_cols = df.columns.tolist()
        try:
            signals = signals_fn(df)
        except Exception as e:
            raise ValueError(
                f"Strategy code referenced column '{e}' which doesn't exist. "
                f"Available columns: {available_cols}. "
                "The LLM likely hallucinated a ticker-prefixed column name "
                "like 'AAPL_Close' instead of 'Close'."
            ) from e
        price = df.set_index("Date")["Close"]

        entries = signals == 1
        exits = signals == -1

        pf = vbt.Portfolio.from_signals(
            price,
            entries,
            exits,
            init_cash=init_cash / len(features),
            freq="D",
        )
        stats = pf.stats()

        per_ticker_results[ticker] = {
            "total_return": float(stats.get("Total Return [%]", 0)),
            "sharpe_ratio": float(stats.get("Sharpe Ratio", 0)),
            "max_drawdown": float(stats.get("Max Drawdown [%]", 0)),
            "win_rate": float(stats.get("Win Rate [%]", 0)),
            "num_trades": int(stats.get("Num Trades", 0)),
            "final_value": float(pf.final_value()),
        }

    combined = {"per_ticker": per_ticker_results}
    returns = [r["total_return"] for r in per_ticker_results.values()]
    sharpes = [r["sharpe_ratio"] for r in per_ticker_results.values()]
    drawdowns = [r["max_drawdown"] for r in per_ticker_results.values()]

    combined["avg_return"] = sum(returns) / len(returns) if returns else 0
    combined["avg_sharpe"] = sum(sharpes) / len(sharpes) if sharpes else 0
    combined["avg_max_drawdown"] = (
        sum(drawdowns) / len(drawdowns) if drawdowns else 0
    )
    combined["total_final_value"] = sum(
        r["final_value"] for r in per_ticker_results.values()
    )
    combined["total_return_pct"] = (
        (combined["total_final_value"] / init_cash) - 1
    ) * 100

    return combined
