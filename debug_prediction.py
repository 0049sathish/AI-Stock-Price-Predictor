# debug_prediction.py
import joblib, os, pandas as pd, numpy as np

TICKER = "AAPL"
PROCESSED = f"data/processed/{TICKER}_processed.csv"
MODEL = f"models/{TICKER}_rf.pkl"

print("Paths exist:", os.path.exists(PROCESSED), os.path.exists(MODEL))

# Load processed data
df = pd.read_csv(PROCESSED, index_col=0, parse_dates=True)
print("Processed shape:", df.shape)
print("Processed columns:", df.columns.tolist())

# Show last few rows
print("\nLast 5 rows (Close, target_close_next):")
print(df[["Close", "target_close_next"]].tail(5))

# Last row used by API
last_row = df.iloc[-1:]
print("\nLast row (all numeric columns):")
print(last_row.select_dtypes(include=[np.number]).T)

# Prepare X exactly like API
X = last_row.select_dtypes(include=[np.number]).drop(columns=["target_close_next"], errors=False)
print("\nX columns used for prediction:")
print(X.columns.tolist())

# Load model
model = joblib.load(MODEL)
print("\nLoaded model type:", type(model))
if hasattr(model, "feature_names_in_"):
    print("Model.feature_names_in_ sample (first 50):", model.feature_names_in_[:50])

# Align features if model expects feature_names_in_
if hasattr(model, "feature_names_in_"):
    cols = list(model.feature_names_in_)
    for c in cols:
        if c not in X.columns:
            X[c] = 0.0
    X = X[cols]

print("\nFinal X shape:", X.shape)
print("Final X preview:", X.iloc[0].to_dict())

# Predict
pred = model.predict(X)[0]
print("\nModel prediction:", pred)

# Naive baseline: predict next_close = today's close
today_close = float(last_row["Close"].values[0])
print("Today's close (naive next-day):", today_close)
print("Model - Naive difference:", pred - today_close)
