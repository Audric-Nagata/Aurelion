from tools.market_data import fetch_market_data


def test_fetch_single_ticker():
    data = fetch_market_data(["AAPL"], "2024-01-01", "2024-01-10")
    assert "AAPL" in data
    records = data["AAPL"]
    assert len(records) > 0
    assert "Date" in records[0]
    assert "Close" in records[0]
    assert "ticker" in records[0]
    assert records[0]["ticker"] == "AAPL"


def test_fetch_multiple_tickers():
    data = fetch_market_data(
        ["AAPL", "MSFT"], "2024-01-01", "2024-01-10"
    )
    assert "AAPL" in data
    assert "MSFT" in data
    assert len(data["AAPL"]) > 0
    assert len(data["MSFT"]) > 0
