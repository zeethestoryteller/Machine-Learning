# Topic 2: Decomposition

### The big idea
Decomposition takes a raw series and splits it into the three components we just discussed (trend, seasonal, residual) so you can *see* and *analyze* each one separately. It's diagnostic — you do this early to understand what you're dealing with before choosing a model.

There are two common approaches: **classical decomposition** and **STL**.

### Classical decomposition

```python
from statsmodels.tsa.seasonal import seasonal_decompose, STL

# Classical decomposition
decomp = seasonal_decompose(ts['value'], model='additive', period=12)
trend    = decomp.trend
seasonal = decomp.seasonal
residual = decomp.resid
```

How it actually works under the hood:
- **Trend** is estimated using a moving average (a rolling window average, roughly of length = `period`). This is why the trend series has `NaN`s at the start and end — a centered moving average needs data on both sides.
- **Seasonal** is estimated by averaging the detrended values for each position in the cycle (e.g., average of all Januaries, all Februaries, etc.) — so it's forced to repeat identically every cycle.
- **Residual** is just whatever's left: `value - trend - seasonal` (additive) or `value / (trend * seasonal)` (multiplicative).

**Limitation to know**: because the seasonal component is a fixed repeating pattern and the trend is a simple moving average, classical decomposition struggles if the seasonal pattern itself evolves over time, or if there are outliers.

### STL (Seasonal-Trend decomposition using LOESS)

```python
# STL: more robust
stl = STL(ts['value'], period=12, seasonal=7)
result = stl.fit()

trend_stl    = result.trend
seasonal_stl = result.seasonal
residual_stl = result.resid
result.plot()
```

STL is generally preferred in practice because:
- It uses **LOESS** (locally weighted regression) instead of a simple moving average, so the trend can flex more naturally.
- Critically, it allows the **seasonal component to change gradually over time** rather than being rigidly identical every cycle — closer to how real-world seasonality behaves (e.g., holiday shopping patterns shift slightly year to year).
- It's more robust to outliers.

The `seasonal=7` parameter controls the smoothing window for the seasonal component (must be odd) — larger values make the seasonal pattern more rigid/stable across cycles, smaller values let it adapt faster.

### Additive vs Multiplicative — how to choose

| Model | Formula | Use when |
|---|---|---|
| Additive | Y = T + S + R | Seasonal swings stay roughly the same size regardless of the trend level |
| Multiplicative | Y = T × S × R | Seasonal swings grow/shrink proportionally as the trend level grows |

**Visual test**: plot your data. If the "wiggles" (seasonal amplitude) stay a constant height as the series trends upward, it's additive. If the wiggles get *bigger* as the series grows (e.g., retail sales — December spikes get larger in absolute terms as the business grows), it's multiplicative.

```python
# Log transform converts multiplicative → additive
ts['log_value'] = np.log(ts['value'])
```

This trick is worth understanding, not just memorizing: if `Y = T × S × R`, then `log(Y) = log(T) + log(S) + log(R)`. Taking the log turns a multiplicative relationship into an additive one — so you can apply additive tools (including plain ARIMA, which assumes additive structure) to data that's actually multiplicative in its raw form. You just remember to exponentiate your forecasts back at the end.

### Quick intuition check
If you're modeling airline passenger counts, where seasonal peaks (summer travel) get noticeably larger year after year as air travel grows in popularity — additive or multiplicative? *(Answer: multiplicative — swing size scales with the trend level.)*

Ready for **Topic 3: Stationarity** next?
