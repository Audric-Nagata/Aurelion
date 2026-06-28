import numpy as np


def compute_risk_metrics(returns: list[float]) -> dict:
    """Compute VaR, CVaR, Sortino ratio, and volatility clustering."""
    arr = np.array(returns)
    mu = np.mean(arr)
    sigma = np.std(arr, ddof=1)

    if sigma == 0 or len(arr) < 2:
        return {"var_95": 0, "var_99": 0, "cvar_95": 0, "sortino": 0, "annualized_vol": 0}

    var_95 = np.percentile(arr, 5)
    var_99 = np.percentile(arr, 1)
    cvar_95 = arr[arr <= var_95].mean() if np.any(arr <= var_95) else var_95

    downside = arr[arr < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1.0
    sortino = mu / downside_std * np.sqrt(252)

    annualized_vol = sigma * np.sqrt(252)

    return {
        "var_95": round(float(var_95), 4),
        "var_99": round(float(var_99), 4),
        "cvar_95": round(float(cvar_95), 4),
        "sortino": round(float(sortino), 4),
        "annualized_vol": round(float(annualized_vol), 4),
    }
