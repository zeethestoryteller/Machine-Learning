# Topic 1: Time Series Fundamentals

### The big idea
Time series data is fundamentally different from regular tabular data because **order matters**. If you shuffle rows in a normal ML dataset, nothing breaks. If you shuffle a time series, you destroy the entire signal — trends, seasonality, and autocorrelation all depend on things happening in sequence.

This has a huge practical consequence: **you can never use future data to predict the past**. Every technique in this deck — splitting, feature engineering, cross-validation — exists specifically to respect this constraint.

### The four components of a time series
Any time series can conceptually be broken into:

| Component | What it is | How to remove it |
|---|---|---|
| **Trend** | Long-term direction (up/down over years) | Differencing, linear detrend |
| **Seasonality** | Fixed, repeating pattern (e.g. every 12 months, every 7 days) | Seasonal differencing, STL |
| **Cyclical** | Irregular, non-fixed-period oscillation (e.g. business cycles) | Harder — no fixed period to subtract |
| **Residual** | Whatever noise is left over | This is what's "unexplained" |

The key distinction people often miss: **seasonality has a fixed, known period** (like "every December" or "every Monday"), while **cyclical** patterns (like economic boom/bust cycles) don't have a fixed length — that's why they're harder to model.

### The code: loading and preparing a time series

```python
import pandas as pd, numpy as np

ts = pd.read_csv('timeseries.csv', parse_dates=['date'], index_col='date')
ts = ts.asfreq('D')            # ensure daily frequency
ts = ts.sort_index()
ts = ts.ffill()                # fill small gaps (OK for short runs)

# Set frequency for statsmodels
ts.index.freq = pd.infer_freq(ts.index)
print(ts.index.freq)
```

Let's break this down line by line, because each line is solving a specific, common problem:

1. **`parse_dates=['date'], index_col='date'`** — makes the date column an actual `DatetimeIndex`, not just strings. Almost every time series function in `statsmodels`/`pandas` expects this.

2. **`ts.asfreq('D')`** — this is the line beginners skip and then get bugs later. Your data might have *gaps* (e.g. missing a day). `asfreq('D')` forces the index to have one row per calendar day, **inserting `NaN` rows for missing dates**. Without this, models like SARIMA don't know how to space out lags correctly — they just see "next row" as "next time step," even if a week is actually missing.

3. **`sort_index()`** — guarantees chronological order. Never assume your CSV was saved in order.

4. **`ffill()`** — forward-fill: fill each `NaN` with the *previous* valid value. This is a deliberate, causal choice — it only uses past information, never future information. (Compare to `bfill()` or interpolation, which could leak future values backward. For time series, always be conscious of this.)

5. **`pd.infer_freq(ts.index)`** — auto-detects the frequency string (like `'D'` for daily, `'M'` for monthly) and assigns it explicitly. Many `statsmodels` functions (ARIMA, SARIMA, seasonal_decompose) **require** `.index.freq` to be set — they'll throw warnings or behave incorrectly otherwise.

### Quick check on understanding
Given monthly data `[10, 12, 15, 13, 20, 22]`, which component would you attribute the overall upward drift to, and which would you attribute a repeating dip every 3 months to?

Ready for **Topic 2: Decomposition** (classical vs STL) next, or do you want to sit with this one a bit more?
