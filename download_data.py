import os
import yfinance as yf

TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "META", "NFLX"]

def download_ticker(ticker, start="2010-01-01", end=None, out_path="data/raw"):
    os.makedirs(out_path, exist_ok=True)
    print(f"Downloading: {ticker}")
    df = yf.download(ticker, start=start, end=end, progress=False)

    if df.empty:
        print(f"!! WARNING: No data downloaded for {ticker}")
        return

    fname = os.path.join(out_path, f"{ticker}.csv")
    df.to_csv(fname)
    print(f"Saved {fname}")

if __name__ == "__main__":
    for ticker in TICKERS:
        download_ticker(ticker)
    print("\n🎯 DONE! All ticker data downloaded.")
