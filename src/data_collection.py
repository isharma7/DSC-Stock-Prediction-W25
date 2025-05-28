import yfinance as yf
import pandas as pd
import os

def download_stock(tickers=["AAPL", "TSLA", "SPY"], start="2020-01-01", end="2024-12-31"):
    os.makedirs("../data", exist_ok=True)

    for ticker in tickers:
        print(f"downloading {ticker}...")
        df = yf.download(ticker, start=start, end=end)

        if df.empty:
            print(f"no data for {ticker}")
            continue

        df.to_csv(f"../data/{ticker}.csv")
        print(f"{ticker} data saved to ../data/{ticker}.csv")

if __name__ == "__main__":
    download_stock(["AAPL", "TSLA", "SPY"])