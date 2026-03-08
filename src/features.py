# src/features.py
import os
import numpy as np
import pandas as pd

from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands

# You can reuse the same tickers list as in your download script
TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "META", "NFLX"]

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"


def load_raw(ticker: str = "AAPL", raw_path: str = RAW_DIR) -> pd.DataFrame:
    """
    Load raw OHLCV data for a given ticker from CSV.
    Expects files like data/raw/AAPL.csv with a Date index.
    """
    fname = os.path.join(raw_path, f"{ticker}.csv")
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Raw file not found for {ticker}: {fname}")

    # yfinance .to_csv writes Date as first column
    df = pd.read_csv(fname, parse_dates=True, index_col=0)
    return df


def _flatten_columns_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns (if any) and make sure column names are plain."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def _ensure_scalar_numeric_column(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Safely extract scalar numeric values from a column that might contain
    arrays/tuples/objects. Returns a numeric Series.
    """
    if col not in df.columns:
        # create column of NaNs so code that expects the column doesn't crash
        return pd.Series(np.nan, index=df.index, name=col)

    series = df[col]

    def _to_scalar(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            return x[0] if len(x) > 0 else np.nan
        return x

    scalar = series.map(_to_scalar)
    scalar = pd.to_numeric(scalar, errors="coerce")
    scalar.index = df.index
    scalar.name = col
    return scalar


def engineer_features(df: pd.DataFrame, lags: int = 5) -> pd.DataFrame:
    """
    df: raw dataframe from yfinance (or CSV)
    Returns a dataframe with engineered features and target 'target_close_next'.
    """
    df = df.copy()
    df = _flatten_columns_if_needed(df)

    # Ensure core OHLCV columns exist and are numeric
    df["Close"] = _ensure_scalar_numeric_column(df, "Close")
    df["Open"] = _ensure_scalar_numeric_column(df, "Open")
    df["High"] = _ensure_scalar_numeric_column(df, "High")
    df["Low"] = _ensure_scalar_numeric_column(df, "Low")
    df["Volume"] = _ensure_scalar_numeric_column(df, "Volume")

    # Drop rows where Close is NaN (can't compute indicators)
    df = df.dropna(subset=["Close"])

    # Basic returns
    df["return"] = df["Close"].pct_change()

    # Lag features of close & returns
    for lag in range(1, lags + 1):
        df[f"lag_close_{lag}"] = df["Close"].shift(lag)
        df[f"lag_return_{lag}"] = df["return"].shift(lag)

    # Moving averages
    df["sma_10"] = df["Close"].rolling(window=10).mean()
    df["ema_10"] = df["Close"].ewm(span=10, adjust=False).mean()

    # RSI
    try:
        rsi = RSIIndicator(df["Close"], window=14)
        df["rsi_14"] = rsi.rsi()
    except Exception:
        df["rsi_14"] = np.nan

    # Bollinger bands
    try:
        bb = BollingerBands(df["Close"], window=20, window_dev=2)
        df["bb_hband"] = bb.bollinger_hband()
        df["bb_lband"] = bb.bollinger_lband()
    except Exception:
        df["bb_hband"] = np.nan
        df["bb_lband"] = np.nan

    # Volume change
    df["vol_change"] = df["Volume"].pct_change()

    # Target: next day's close
    df["target_close_next"] = df["Close"].shift(-1)

    # Drop rows with NaN created by rolling/lag
    df = df.dropna()

    return df


def save_processed(df: pd.DataFrame, ticker: str = "AAPL", out_path: str = PROCESSED_DIR) -> str:
    os.makedirs(out_path, exist_ok=True)
    fname = os.path.join(out_path, f"{ticker}_processed.csv")
    df.to_csv(fname)
    print(f"Saved processed: {fname}")
    return fname


if __name__ == "__main__":
    # Process all tickers in the list
    for t in TICKERS:
        try:
            raw = load_raw(t)
            fe = engineer_features(raw)
            save_processed(fe, ticker=t)
        except Exception as e:
            print(f"⚠ Failed processing {t}: {e}")
