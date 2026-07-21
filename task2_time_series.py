"""
Task 2: Time Series Analysis
Dataset: AAPL daily stock prices (2014-2017)
Tools: Python, pandas, matplotlib, statsmodels
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# ---------------------------------------------------------
# 1. Load and prepare the data
# ---------------------------------------------------------
df = pd.read_csv("aapl_prices.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").set_index("date")

print("=" * 50)
print("STEP 1: Overview")
print("=" * 50)
print(df[["close"]].describe())

# ---------------------------------------------------------
# 2. Plot time-series data and identify patterns
# ---------------------------------------------------------
plt.figure(figsize=(12, 5))
plt.plot(df.index, df["close"], color="#1f77b4")
plt.title("AAPL Closing Price (2014-2017)")
plt.xlabel("Date")
plt.ylabel("Close Price ($)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("aapl_price_trend.png", dpi=150)
plt.close()
print("\nSaved aapl_price_trend.png — shows a clear long-term upward trend.")

# ---------------------------------------------------------
# 3. Decompose into trend, seasonality, residuals
# ---------------------------------------------------------
# Use business-day frequency and forward-fill any gaps (holidays)
ts = df["close"].asfreq("B").ffill()

decomposition = seasonal_decompose(ts, model="additive", period=252)  # ~252 trading days/year

fig = decomposition.plot()
fig.set_size_inches(12, 8)
fig.tight_layout()
fig.savefig("aapl_decomposition.png", dpi=150)
plt.close()
print("Saved aapl_decomposition.png (trend, seasonal, residual components).")

# ---------------------------------------------------------
# 4. Moving average smoothing
# ---------------------------------------------------------
df["MA_30"] = df["close"].rolling(window=30).mean()
df["MA_90"] = df["close"].rolling(window=90).mean()

plt.figure(figsize=(12, 5))
plt.plot(df.index, df["close"], label="Daily Close", alpha=0.4)
plt.plot(df.index, df["MA_30"], label="30-Day Moving Average", linewidth=2)
plt.plot(df.index, df["MA_90"], label="90-Day Moving Average", linewidth=2)
plt.title("AAPL Closing Price with Moving Averages")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("aapl_moving_averages.png", dpi=150)
plt.close()
print("Saved aapl_moving_averages.png")

print("\nDone. All plots exported.")
