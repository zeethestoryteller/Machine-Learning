# Topic 3: Stationarity

### Why this matters so much
Most classical forecasting models (ARIMA in particular) **assume stationarity**. If you feed a non-stationary series into these models without preparing it, your forecasts and confidence intervals will be unreliable — the model is essentially assuming statistical properties that don't actually hold.

### Definition
A series is stationary if it has:
- **Constant mean over time** — no upward/downward drift
- **Constant variance over time** — the "spread" of values doesn't grow or shrink
- **Autocorrelation depends only on lag, not on time** — the relationship between `y_t` and `y_{t-5}` is the same whether you're looking at the start or end of the series

Intuitively: a stationary series looks statistically "the same" no matter which window of time you crop out and examine.

### The Augmented Dickey-Fuller (ADF) test

```
H₀: unit root exists (non-stationary)
H₁: no unit root (stationary)

p-value < 0.05 → reject H₀ → stationary
```

### The trap — pay close attention here
This is the single most common point of confusion in this whole topic, and the PDF calls it out explicitly for good reason:

> **The null hypothesis (H₀) is that the series is non-stationary.**

This is backwards from how most students expect statistical tests to work (usually H₀ = "nothing interesting/no effect"). Here, H₀ = "bad news" (unit root, non-stationary).

So the logic chain is:
- **Low p-value (< 0.05)** → reject H₀ → **conclude the series IS stationary** (good news)
- **High p-value (≥ 0.05)** → fail to reject H₀ → **series is likely non-stationary** (needs fixing)

It's a double negative that trips people up: "low p-value" sounds like it should mean "low confidence in stationarity," but it's the opposite — it means strong evidence *against* the non-stationary hypothesis.

### The code

```python
from statsmodels.tsa.stattools import adfuller, kpss

def test_stationarity(series, name=''):
    result = adfuller(series.dropna())
    print(f"{name} ADF Statistic: {result[0]:.4f}, p-value: {result[1]:.4f}")
    print(f"Stationary: {result[1] < 0.05}")

test_stationarity(ts['value'], 'Original')

# Fix non-stationarity
ts['diff1']    = ts['value'].diff(1)      # remove trend
ts['diff12']   = ts['value'].diff(12)     # remove seasonality
ts['diff1_12'] = ts['diff1'].diff(12)     # both

test_stationarity(ts['diff1'].dropna(), 'First diff')
test_stationarity(ts['diff1_12'].dropna(), 'Seasonal diff')
```

Walking through what each differencing line does:
- **`.diff(1)`** computes `y_t - y_{t-1}` — this removes a **linear trend**. If your series is steadily climbing, differencing once turns "climbing values" into "roughly constant step sizes."
- **`.diff(12)`** computes `y_t - y_{t-12}` — this removes **annual seasonality** in monthly data (period=12) by comparing each point to the same point one full cycle earlier.
- **`.diff1.diff(12)`** applies both — first remove trend, then remove what's left of the seasonal pattern from the *already-detrended* series.

Note: `adfuller` returns a tuple; index `[0]` is the test statistic, `[1]` is the p-value. `.dropna()` is required because differencing always introduces `NaN`s at the start (there's no `y_{t-1}` for the very first observation).

### Transformations cheat sheet

| Problem | Fix |
|---|---|
| Trend | First differencing (d=1) |
| Seasonal pattern | Seasonal differencing (D=1, period=12) |
| Growing variance | Log transform or Box-Cox |
| Structural break | Segment the data, or use an intervention model |

Note the distinction: **differencing** fixes mean-related non-stationarity (trend, seasonality), while **log/Box-Cox transforms** fix variance-related non-stationarity (the spread growing over time). These are different problems needing different fixes — you might need both.

There's also `kpss` imported but not used in the snippet — worth knowing: **KPSS has the opposite null hypothesis** (H₀ = stationary). Practitioners often run *both* ADF and KPSS together, because relying on one test alone can give misleading conclusions in edge cases (e.g., series that are neither purely trend-stationary nor purely difference-stationary).

### Quick check
If ADF gives you p = 0.23 on your raw series, is it stationary? What should you do next?

*(Answer: No — p ≥ 0.05 means we fail to reject H₀, so it's non-stationary. Next step: apply differencing, then re-run the ADF test on the differenced series.)*

Ready for **Topic 4: ACF & PACF Interpretation**?
