import yfinance as yf
import pandas as pd
import os

TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "META", "NFLX"]
DATA_DIR = "data/raw"

def download_ticker(ticker, start="2010-01-01", end=None):
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Downloading: {ticker}")
    df = yf.download(ticker, start=start, end=end, progress=False)

    if df.empty:
        print(f"⚠ No data downloaded for {ticker}")
        return

    filepath = os.path.join(DATA_DIR, f"{ticker}.csv")
    df.to_csv(filepath)
    print(f"✔ Saved {filepath}")

if __name__ == "__main__":
    for t in TICKERS:
        download_ticker(t)
