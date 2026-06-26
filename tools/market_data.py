import yfinance as yf


def fetch_market_data(tickers: list[str], start: str, end: str) -> dict:
    """Fetch OHLCV data from yfinance for given tickers and date range."""
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)

    if len(tickers) == 1:
        ticker = tickers[0]
        df = data.reset_index()
        df["ticker"] = ticker
        return {ticker: df.to_dict(orient="records")}

    result = {}
    for ticker in tickers:
        if ticker in data.columns.get_level_values(1):
            df = data.xs(ticker, axis=1, level=1).reset_index()
            df["ticker"] = ticker
            result[ticker] = df.to_dict(orient="records")
    return result
