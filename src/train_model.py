# src/train_model.py

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Same list you used in download script / features.py
TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "META", "NFLX"]


def data_path_for(ticker: str) -> str:
    return os.path.join(DATA_DIR, f"{ticker}_processed.csv")


def load_data(ticker: str):
    path = data_path_for(ticker)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed file not found for {ticker}: {path}")

    df = pd.read_csv(path)

    # ensure Date index if present
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
    else:
        # sometimes first col is the date index when saved
        first_col = df.columns[0]
        try:
            df.index = pd.to_datetime(df[first_col])
            df = df.drop(columns=[first_col])
        except Exception:
            pass

    if "target_close_next" not in df.columns:
        raise ValueError(
            f"Column 'target_close_next' missing in processed data for {ticker}!"
        )

    y = df["target_close_next"]
    X = df.drop(columns=["target_close_next"])
    return X, y


# ---------------- classical models ----------------
def train_rf(X, y):
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def train_lr(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model


# ---------------- "deep" models using MLP ----------------
def train_lstm_like(X, y):
    """
    Stand-in for LSTM using a feed-forward MLP.
    We save it as *_lstm.pkl so backend & UI can call it 'LSTM'.
    """
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        random_state=42,
        max_iter=200,
    )
    model.fit(X, y)
    return model


def train_gru_like(X, y):
    """
    Stand-in for GRU using a slightly different MLP.
    Saved as *_gru.pkl.
    """
    model = MLPRegressor(
        hidden_layer_sizes=(64, 64),
        activation="tanh",
        solver="adam",
        random_state=42,
        max_iter=200,
    )
    model.fit(X, y)
    return model


def evaluate(model, X, y, label="train"):
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    print(f"[{label}] R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")


# ---------------- main training loop ----------------
if __name__ == "__main__":
    print("PROJECT_ROOT:", PROJECT_ROOT)

    for ticker in TICKERS:
        print("=" * 70)
        print(f"Training models for {ticker} ...")
        try:
            X, y = load_data(ticker)
        except Exception as e:
            print(f"⚠ Skipping {ticker}: {e}")
            continue

        print("Data shape:", X.shape, y.shape)

        # Random Forest
        rf_model = train_rf(X, y)
        evaluate(rf_model, X, y, label=f"{ticker} RF train")
        rf_path = os.path.join(MODEL_DIR, f"{ticker}_rf.pkl")
        joblib.dump(rf_model, rf_path)
        print("Saved Random Forest ->", rf_path)

        # Linear Regression
        lr_model = train_lr(X, y)
        evaluate(lr_model, X, y, label=f"{ticker} LR train")
        lr_path = os.path.join(MODEL_DIR, f"{ticker}_lr.pkl")
        joblib.dump(lr_model, lr_path)
        print("Saved Linear Regression ->", lr_path)

        # "LSTM"
        lstm_model = train_lstm_like(X, y)
        evaluate(lstm_model, X, y, label=f"{ticker} LSTM-like train")
        lstm_path = os.path.join(MODEL_DIR, f"{ticker}_lstm.pkl")
        joblib.dump(lstm_model, lstm_path)
        print("Saved LSTM-like ->", lstm_path)

        # "GRU"
        gru_model = train_gru_like(X, y)
        evaluate(gru_model, X, y, label=f"{ticker} GRU-like train")
        gru_path = os.path.join(MODEL_DIR, f"{ticker}_gru.pkl")
        joblib.dump(gru_model, gru_path)
        print("Saved GRU-like ->", gru_path)
