# streamlit_app.py
# Streamlit frontend that calls your existing FastAPI backend (http://localhost:8002)

import os
import sys
from datetime import datetime, date
from typing import Dict, List

import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
API_URL = "http://localhost:8002"   # FastAPI backend

# Labels for model selector
MODEL_LABELS = {
    "rf": "Random Forest",
    "lr": "Linear Regression",
    "lstm": "LSTM",
    "gru": "GRU",
}

# -------------------------------------------------------------------
# Backend helpers
# -------------------------------------------------------------------
@st.cache_data(ttl=300)
def api_health() -> Dict:
    """Get available tickers + models from the backend /health endpoint."""
    resp = requests.get(f"{API_URL}/health", timeout=10)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=300)
def api_history(ticker: str, days: int = 180) -> pd.DataFrame:
    """Call /history and return a clean DataFrame indexed by Date."""
    params = {"ticker": ticker, "days": days}
    resp = requests.get(f"{API_URL}/history", params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    hist = data.get("history", [])
    if not hist:
        raise ValueError(f"No history returned for {ticker}")

    df = pd.DataFrame(hist)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    # Ensure numeric
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Close"])
    return df


def api_predict(ticker: str, model_name: str) -> Dict:
    """Call /predict and return the JSON."""
    payload = {"ticker": ticker, "model_name": model_name}
    resp = requests.post(f"{API_URL}/predict", json=payload, timeout=20)
    resp.raise_for_status()
    return resp.json()


# -------------------------------------------------------------------
# Plotting helpers
# -------------------------------------------------------------------
def plot_price_history_with_prediction(
    history_df: pd.DataFrame,
    pred_date_str: str,
    pred_value: float,
    ticker: str,
):
    """Candlestick + prediction dot (like your React UI, no bar chart)."""
    pred_dt = datetime.fromisoformat(pred_date_str)

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=history_df.index,
            open=history_df["Open"],
            high=history_df["High"],
            low=history_df["Low"],
            close=history_df["Close"],
            name="Price History",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[pred_dt],
            y=[pred_value],
            mode="markers+text",
            text=["Predicted close"],
            textposition="top center",
            name="Prediction",
        )
    )

    fig.update_layout(
        title=f"{ticker} — Historical Prices & Forecast",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=500,
    )
    return fig


def plot_multi_ticker_history(tickers: List[str]):
    """Line chart of Close prices for several tickers (no bars)."""
    fig = go.Figure()

    for t in tickers:
        try:
            df = api_history(t, days=180)
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["Close"],
                    mode="lines",
                    name=t,
                )
            )
        except Exception as e:
            st.warning(f"Failed to load history for {t}: {type(e).__name__}: {e}")

    fig.update_layout(
        title="Multi-ticker comparison (closing prices)",
        xaxis_title="Date",
        yaxis_title="Closing price",
        template="plotly_dark",
        height=500,
    )
    return fig


# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Stock Price Predictor")

# ---- load backend info ----
try:
    health = api_health()
    models_info: Dict[str, List[str]] = health.get("models", {})
except Exception as e:
    st.error(
        "Could not reach backend at "
        f"{API_URL}. Make sure FastAPI is running.\n\n"
        f"Error: {type(e).__name__}: {e}"
    )
    st.stop()

available_tickers = sorted(models_info.keys())
if not available_tickers:
    st.error("Backend reports no loaded models. Check the models/ folder in FastAPI.")
    st.stop()

# ---- controls ----
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    ticker = st.selectbox("Ticker", available_tickers, index=0)

with col2:
    model_keys_for_ticker = sorted(models_info.get(ticker, []))
    model_key = st.selectbox(
        "Model",
        model_keys_for_ticker,
        format_func=lambda k: MODEL_LABELS.get(k, k.upper()),
    )

with col3:
    do_predict = st.button("Predict", type="primary")

# -------------------------------------------------------------------
# Prediction + price history section
# -------------------------------------------------------------------
if "last_pred" not in st.session_state:
    st.session_state["last_pred"] = None
if "last_error" not in st.session_state:
    st.session_state["last_error"] = None

if do_predict:
    try:
        pred_json = api_predict(ticker, model_key)
        hist_df = api_history(ticker, days=180)

        st.session_state["last_pred"] = {
            "ticker": pred_json["ticker"],
            "model_name": pred_json["model_name"],
            "date": pred_json["date"],
            "value": float(pred_json["predicted_next_close"]),
        }
        st.session_state["last_error"] = None
        st.session_state["last_history"] = hist_df

    except Exception as e:
        st.session_state["last_error"] = f"{type(e).__name__}: {e}"
        st.session_state["last_pred"] = None

# show last result / error
if st.session_state["last_error"]:
    st.error(f"Prediction failed: {st.session_state['last_error']}")

if st.session_state["last_pred"]:
    info = st.session_state["last_pred"]
    ticker = info["ticker"]
    model_name = info["model_name"]
    pred_date_str = info["date"]
    pred_value = info["value"]
    label = MODEL_LABELS.get(model_name, model_name.upper())

    dt_obj = datetime.fromisoformat(pred_date_str)
    st.success(
        f"{ticker} — prediction for {dt_obj.strftime('%d %b %Y')} "
        f"using {label}: {pred_value:.2f}"
    )

    hist_df = st.session_state.get("last_history")
    if hist_df is not None:
        fig_price = plot_price_history_with_prediction(
            hist_df, pred_date_str, pred_value, ticker
        )
        st.plotly_chart(fig_price, use_container_width=True)

# -------------------------------------------------------------------
# Multi-ticker section
# -------------------------------------------------------------------
st.markdown("---")
st.subheader("📊 Multi-ticker comparison (closing prices)")

if st.button("Load multi-ticker chart"):
    fig_multi = plot_multi_ticker_history(available_tickers)
    st.plotly_chart(fig_multi, use_container_width=True)
