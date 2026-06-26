import pandas as pd
import numpy as np


def generate_features(market_data: dict, config: dict | None = None) -> dict:
    """Compute technical indicators and factors from market data."""
    result = {}
    for ticker, records in market_data.items():
        df = pd.DataFrame(records)
        df.sort_values("Date", inplace=True)

        df["returns_1d"] = df["Close"].pct_change()
        df["returns_5d"] = df["Close"].pct_change(5)
        df["returns_20d"] = df["Close"].pct_change(20)

        df["SMA_10"] = df["Close"].rolling(10).mean()
        df["SMA_20"] = df["Close"].rolling(20).mean()
        df["SMA_50"] = df["Close"].rolling(50).mean()
        df["SMA_200"] = df["Close"].rolling(200).mean()

        df["volatility_20d"] = df["returns_1d"].rolling(20).std()
        df["volatility_50d"] = df["returns_1d"].rolling(50).std()

        df["rsi_14"] = _compute_rsi(df["Close"], 14)

        df["volume_sma_20"] = df["Volume"].rolling(20).mean()
        df["volume_ratio"] = df["Volume"] / df["volume_sma_20"]

        df["high_low_pct"] = (df["High"] - df["Low"]) / df["Close"]
        df["close_open_pct"] = (df["Close"] - df["Open"]) / df["Open"]

        result[ticker] = df.dropna().to_dict(orient="records")
    return result


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))
