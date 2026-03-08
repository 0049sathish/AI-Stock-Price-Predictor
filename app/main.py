# app/main.py

import os
import sys
import logging
import traceback
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

# ----- Paths -----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features import engineer_features

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
FAVICON_PATH = os.path.join(PROJECT_ROOT, "app", "static", "favicon.ico")

logger = logging.getLogger("uvicorn.error")
logging.basicConfig(level=logging.INFO)

# Loaded models dictionary: {ticker: {model_name: model}}
AVAILABLE_MODELS: Dict[str, Dict[str, Any]] = {}


# ----- Startup + Shutdown -----
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    On startup, scan models/ for files like AAPL_rf.pkl, MSFT_lr.pkl, ...
    and load them into AVAILABLE_MODELS.
    """
    global AVAILABLE_MODELS
    AVAILABLE_MODELS = {}

    if os.path.isdir(MODEL_DIR):
        for fname in os.listdir(MODEL_DIR):
            if not fname.endswith(".pkl"):
                continue

            base = fname[:-4]  # remove .pkl
            if "_" not in base:
                continue

            # Expect "<TICKER>_<model_name>.pkl"
            ticker_part, model_part = base.split("_", 1)
            ticker = ticker_part.upper()
            model_name = model_part.lower()

            model_path = os.path.join(MODEL_DIR, fname)
            try:
                model = joblib.load(model_path)
                AVAILABLE_MODELS.setdefault(ticker, {})[model_name] = model
                logger.info(f"Loaded model {model_name} for {ticker} from {model_path}")
            except Exception:
                logger.exception(f"Failed to load model file: {model_path}")
    else:
        logger.warning(f"MODEL_DIR does not exist: {MODEL_DIR}")

    logger.info(
        "Loaded models: %s",
        {t: list(m.keys()) for t, m in AVAILABLE_MODELS.items()},
    )

    yield

    AVAILABLE_MODELS = {}
    logger.info("Shutting down - models cleared")


app = FastAPI(title="S_K_Project", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----- Request/Response Models -----
class PredictRequest(BaseModel):
    ticker: str = "AAPL"
    model_name: str = "rf"  # default Random Forest

class PredictResponse(BaseModel):
    ticker: str
    model_name: str
    date: str
    predicted_next_close: float 

class ModelPrediction(BaseModel):
    model_name: str
    predicted_next_close: float


class CompareRequest(BaseModel):
    ticker: str = "AAPL"


class CompareResponse(BaseModel):
    ticker: str
    date: str
    results: List[ModelPrediction]


# ----- Utils -----
def fetch_recent_data(ticker: str = "AAPL", days: int = 120):
    """
    Fetch ONLY live data from yfinance.
    No local CSV fallback – if this fails, we raise an error.
    """
    try:
        df = yf.download(
            ticker,
            period=f"{days}d",   # or "6mo" if you prefer a fixed window
            interval="1d",
            progress=False,
        )
    except Exception as e:
        logger.exception(f"yfinance download error for {ticker}: {e}")
        raise ValueError(f"Live data download failed for {ticker}")

    if df is None or df.empty:
        raise ValueError(f"No data returned from yfinance for {ticker}")

    return df




def _short_tb(e: Exception):
    return traceback.format_exc()[:1000]


def _get_model_for(ticker: str, model_name: str):
    """
    Return the model object for a specific ticker + model_name.
    Raises HTTPException if not available.
    """
    global AVAILABLE_MODELS

    ticker = ticker.upper()
    model_name = model_name.lower()

    if ticker not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"No models loaded for ticker {ticker}. "
            f"Available tickers: {list(AVAILABLE_MODELS.keys())}",
        )

    models_for_ticker = AVAILABLE_MODELS[ticker]
    if model_name not in models_for_ticker:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' not available for {ticker}. "
            f"Available: {list(models_for_ticker.keys())}",
        )

    return models_for_ticker[model_name]


# ----- API Endpoints -----
@app.get("/")
def root():
    return {"status": "ok", "message": "Stock API is live!"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models": {t: list(m.keys()) for t, m in AVAILABLE_MODELS.items()},
    }


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    if os.path.exists(FAVICON_PATH):
        return FileResponse(FAVICON_PATH, media_type="image/x-icon")
    raise HTTPException(status_code=404, detail="favicon not found")


@app.get("/history")
def get_history(ticker: str = "AAPL", days: int = 120):
    try:
        df = fetch_recent_data(ticker, days)

        # ---- flatten columns if yfinance returns MultiIndex ----
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                c[0] if isinstance(c, tuple) else str(c) for c in df.columns
            ]

        # make Date a normal column
        df = df.reset_index()
        df["Date"] = df["Date"].astype(str)

        # build a clean list[dict] with only simple keys & float values
        history = []
        for _, row in df.iterrows():
            history.append(
                {
                    "Date": row["Date"],
                    "Open": float(row["Open"]),
                    "High": float(row["High"]),
                    "Low": float(row["Low"]),
                    "Close": float(row["Close"]),
                }
            )

        return {"ticker": ticker.upper(), "history": history}

    except Exception as e:
        logger.exception("History fetch failed for %s", ticker)
        raise HTTPException(status_code=500, detail=f"History fetch failed: {e}")



@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    ticker = (req.ticker or "AAPL").upper()
    model_name = (req.model_name or "rf").lower()

    # use helper to get the right model for this ticker
    model = _get_model_for(ticker, model_name)

    # 1) fetch live data + engineer features
    try:
        raw = fetch_recent_data(ticker)
        df = engineer_features(raw, lags=5)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Feature engineering failed: {e}"
        )

    if df is None or df.empty:
        raise HTTPException(
            status_code=500, detail="No rows after feature engineering"
        )

    last_row = df.iloc[-1:]
    X = last_row.select_dtypes(include=[np.number]).copy()
    X = X.drop(columns=["target_close_next"], errors=False)

    # 2) feature alignment
    if hasattr(model, "feature_names_in_"):
        for col in model.feature_names_in_:
            if col not in X.columns:
                X[col] = 0.0
        X = X[model.feature_names_in_]

    # 3) prediction
    try:
        pred = float(model.predict(X)[0])
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction error: {e}\n{_short_tb(e)}"
        )

    # 4) forecast date = next trading day from *today*, not from index
    from datetime import date, timedelta

    today = date.today()
    # next trading day (skips weekend)
    if today.weekday() == 4:      # Friday -> Monday
        forecast_date = today + timedelta(days=3)
    elif today.weekday() >= 5:    # Saturday/Sunday -> next Monday
        days_ahead = 7 - today.weekday()
        forecast_date = today + timedelta(days=days_ahead)
    else:                         # Mon–Thu -> next day
        forecast_date = today + timedelta(days=1)

    date_str = forecast_date.isoformat()

    return PredictResponse(
        ticker=ticker,
        model_name=model_name,
        date=date_str,
        predicted_next_close=pred,
    )



@app.post("/compare_models", response_model=CompareResponse)
def compare_models(req: CompareRequest):
    ticker = (req.ticker or "AAPL").upper()

    if ticker not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"No models loaded for ticker {ticker}. "
                   f"Available tickers: {list(AVAILABLE_MODELS.keys())}",
        )

    # 1) fetch + features
    try:
        raw = fetch_recent_data(ticker)
        df = engineer_features(raw, lags=5)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Feature engineering failed: {e}"
        )

    if df is None or df.empty:
        raise HTTPException(
            status_code=500, detail="No rows after feature engineering"
        )

    last_row = df.iloc[-1:]
    base_X = last_row.select_dtypes(include=[np.number]).copy()
    base_X = base_X.drop(columns=["target_close_next"], errors=False)

    # 2) forecast date = next trading day from today
    from datetime import date, timedelta

    today = date.today()
    if today.weekday() == 4:      # Friday -> Monday
        forecast_date = today + timedelta(days=3)
    elif today.weekday() >= 5:    # Sat/Sun -> next Monday
        days_ahead = 7 - today.weekday()
        forecast_date = today + timedelta(days=days_ahead)
    else:                         # Mon–Thu -> next day
        forecast_date = today + timedelta(days=1)

    date_str = forecast_date.isoformat()

    # 3) run all models for this ticker
    results: List[ModelPrediction] = []

    for name, model in AVAILABLE_MODELS[ticker].items():
        X = base_X.copy()
        if hasattr(model, "feature_names_in_"):
            for col in model.feature_names_in_:
                if col not in X.columns:
                    X[col] = 0.0
            X = X[model.feature_names_in_]

        try:
            pred = float(model.predict(X)[0])
            results.append(
                ModelPrediction(model_name=name, predicted_next_close=pred)
            )
        except Exception as e:
            logger.exception(f"Prediction failed for {ticker} model {name}: {e}")

    if not results:
        raise HTTPException(status_code=500, detail="All model predictions failed")

    return CompareResponse(ticker=ticker, date=date_str, results=results)



if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8002, reload=True)
