import React, { useState } from "react";
import Plot from "react-plotly.js";
import "./App.css";

// backend URL
const API_URL = "http://localhost:8002";

const TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "META", "NFLX"];

const MODELS = [
  { value: "rf", label: "Random Forest" },
  { value: "lr", label: "Linear Regression" },
  { value: "lstm", label: "LSTM" }, 
  { value: "gru", label: "GRU" },  
];

// nice date text like "Nov 26, 2025"
function formatDate(dateStr) {
  const d = new Date(dateStr);
  if (Number.isNaN(d.getTime())) return dateStr;
  return d.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
  });
}

function App() {
  const [ticker, setTicker] = useState("AAPL");
  const [modelName, setModelName] = useState("rf");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState(null);

  // model comparison
  const [compareResults, setCompareResults] = useState([]);
  const [compareError, setCompareError] = useState(null);
  const [compareLoading, setCompareLoading] = useState(false);

  // multi-ticker comparison
  const [multiTickerData, setMultiTickerData] = useState([]);
  const [multiTickerError, setMultiTickerError] = useState(null);
  const [multiTickerLoading, setMultiTickerLoading] = useState(false);

  async function fetchHistory(tickerSymbol) {
    try {
      const res = await fetch(`${API_URL}/history?ticker=${tickerSymbol}`);
      const data = await res.json();
      if (!res.ok) {
        console.error("History error:", data);
        setHistory([]);
        return;
      }
      setHistory(data.history);
    } catch (err) {
      console.error("fetchHistory error:", err);
      setHistory([]);
    }
  }

  async function handlePredict(e) {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker, model_name: modelName }),
      });

      const data = await res.json();
      if (!res.ok) {
        console.error("API error:", data);
        throw new Error(data.detail || "API error");
      }

      setResult(data);
      await fetchHistory(ticker);
    } catch (err) {
      console.error("Fetch error:", err);
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  }

  async function handleCompareModels() {
    setCompareLoading(true);
    setCompareError(null);
    setCompareResults([]);

    try {
      const res = await fetch(`${API_URL}/compare_models`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker }),
      });

      const data = await res.json();
      if (!res.ok) {
        console.error("Compare error:", data);
        throw new Error(data.detail || "Compare API error");
      }

      setCompareResults(data.results);
    } catch (err) {
      console.error("Compare fetch error:", err);
      setCompareError(err.message || String(err));
    } finally {
      setCompareLoading(false);
    }
  }

  async function handleCompareTickers() {
  setMultiTickerLoading(true);
  setMultiTickerError(null);
  setMultiTickerData([]);

  const tickersToCompare = ["AAPL", "MSFT", "GOOGL", "TSLA"];

  const collected = [];

  try {
    for (const t of tickersToCompare) {
      try {
        const res = await fetch(`${API_URL}/history?ticker=${t}`);
        const data = await res.json();
        if (!res.ok) {
          console.error("History error for", t, data);
          continue; // skip this ticker, continue with others
        }

        collected.push({
          ticker: t,
          dates: data.history.map((p) => p.Date),
          closes: data.history.map((p) => p.Close),
        });
      } catch (innerErr) {
        console.error("Network / parse error for", t, innerErr);
        // just skip this ticker
      }
    }

    if (collected.length === 0) {
      setMultiTickerError("No ticker history could be loaded.");
    } else {
      setMultiTickerData(collected);
    }
  } catch (err) {
    console.error("Multi-ticker error:", err);
    setMultiTickerError(err.message || String(err));
  } finally {
    setMultiTickerLoading(false);
  }
}


  // ----- charts -----

  // H1: main history + prediction chart
  const mainChart =
    history.length > 0 && result ? (
      <Plot
        style={{ width: "100%", height: "100%" }}
        config={{ responsive: true }}
        data={[
          {
            x: history.map((p) => p.Date),
            open: history.map((p) => p.Open),
            high: history.map((p) => p.High),
            low: history.map((p) => p.Low),
            close: history.map((p) => p.Close),
            type: "candlestick",
            name: "Price History",
          },
          {
            x: [result.date],
            y: [result.predicted_next_close],
            mode: "markers+text",
            text: ["Predicted close"],
            textposition: "top center",
            name: "Prediction",
          },
        ]}
        layout={{
          title: `${result.ticker} — Historical Prices & Forecast`,
          xaxis: { title: "Date", rangeslider: { visible: false } },
          yaxis: { title: "Price" },
          margin: { l: 50, r: 10, t: 40, b: 40 },
        }}
      />
    ) : null;

  // H3: improved model comparison bar chart
  const compareChart =
    compareResults.length > 0 ? (
      <Plot
        style={{ width: "100%", height: "100%" }}
        config={{ responsive: true }}
        data={[
          {
            x: compareResults.map(
              (r) =>
                MODELS.find((m) => m.value === r.model_name)?.label ||
                r.model_name
            ),
            y: compareResults.map((r) => r.predicted_next_close),
            type: "bar",
            text: compareResults.map((r) => r.predicted_next_close.toFixed(2)),
            textposition: "auto",
            name: "Model predictions",
          },
        ]}
        layout={{
          title: `Model comparison for ${ticker}`,
          xaxis: { title: "Model" },
          yaxis: { title: "Predicted close" },
          margin: { l: 50, r: 10, t: 40, b: 40 },
        }}
      />
    ) : null;

  // H2: multi-ticker comparison line chart
  const multiTickerChart =
    multiTickerData.length > 0 ? (
      <Plot
        style={{ width: "100%", height: "100%" }}
        config={{ responsive: true }}
        data={multiTickerData.map((d) => ({
          x: d.dates,
          y: d.closes,
          type: "scatter",
          mode: "lines",
          name: d.ticker,
        }))}
        layout={{
          title: "Ticker comparison (closing prices)",
          xaxis: { title: "Date" },
          yaxis: { title: "Closing price" },
          margin: { l: 50, r: 10, t: 40, b: 40 },
        }}
      />
    ) : null;

  return (
    <div className="App-root">
      <div className="App-card">
        <h1>Stock Price Predictor</h1>

        <form onSubmit={handlePredict} className="form-row">
          <label>
            Ticker:
            <select
              value={ticker}
              onChange={(e) => setTicker(e.target.value)}
              className="ticker-select"
            >
              {TICKERS.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
          </label>

          <label>
            Model:
            <select
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              className="ticker-select"
            >
              {MODELS.map((m) => (
                <option key={m.value} value={m.value}>
                  {m.label}
                </option>
              ))}
            </select>
          </label>

          <button type="submit" disabled={loading} className="btn-primary">
            {loading ? "Predicting..." : "Predict"}
          </button>

          <button
            type="button"
            onClick={handleCompareModels}
            disabled={compareLoading}
            className="btn-secondary"
          >
            {compareLoading ? "Comparing..." : "Compare models"}
          </button>

          <button
            type="button"
            onClick={handleCompareTickers}
            disabled={multiTickerLoading}
            className="btn-secondary"
          >
            {multiTickerLoading ? "Loading tickers..." : "Compare tickers"}
          </button>
        </form>

        {error && <div className="error-text">Error: {error}</div>}

        {result && !error && (
          <div className="result-text">
            <strong>{result.ticker}</strong> — prediction for{" "}
            {formatDate(result.date)} using{" "}
            <strong>
              {MODELS.find((m) => m.value === modelName)?.label || modelName}
            </strong>
            : <b>{result.predicted_next_close.toFixed(2)}</b>
          </div>
        )}

        <div className="chart-container">{mainChart}</div>

        {compareError && (
          <div className="error-text" style={{ marginTop: 16 }}>
            Compare error: {compareError}
          </div>
        )}
        {compareChart && (
          <div className="chart-container" style={{ marginTop: 24 }}>
            {compareChart}
          </div>
        )}

        {multiTickerError && (
          <div className="error-text" style={{ marginTop: 16 }}>
            Ticker compare error: {multiTickerError}
          </div>
        )}
        {multiTickerChart && (
          <div className="chart-container" style={{ marginTop: 24 }}>
            {multiTickerChart}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
