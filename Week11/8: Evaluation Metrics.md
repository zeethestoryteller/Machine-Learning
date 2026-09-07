# Topic 8: Evaluation Metrics

### The metrics table

| Metric | Formula | Notes |
|---|---|---|
| **RMSE** | √mean((y−ŷ)²) | Penalizes large errors; scale-dependent |
| **MAE** | mean(\|y−ŷ\|) | Scale-dependent, less sensitive to outliers than RMSE |
| **MAPE** | 100·mean(\|(y−ŷ)/y\|) | Scale-independent (%), but breaks when y is near 0 |
| **sMAPE** | 200·mean(\|y−ŷ\|/(\|y\|+\|ŷ\|)) | "Symmetric" version, more stable near 0 |
| **MASE** | MAE / MAE_naive | > 1 = worse than naive baseline |

### Understanding what each one is really telling you

**RMSE vs MAE — same units, different sensitivity**
Both are in the original units of your data (e.g., dollars, degrees). The difference is that RMSE *squares* the errors before averaging, which means large errors get punished disproportionately more than small ones. If you care a lot about avoiding occasional big misses (e.g., stockout predictions), RMSE is more informative. If you want a metric that treats all errors proportionally, MAE is more robust to outliers.

**MAPE — intuitive but fragile**
MAPE gives you a percentage, which is nice because it's scale-independent (you can compare MAPE across a series in the thousands vs a series in the single digits). The catch: if `y` is ever zero or close to zero, you get division by a tiny number and the metric explodes or becomes undefined. Never use MAPE on data that crosses or approaches zero (e.g., temperature in Celsius, net profit/loss).

**sMAPE — a partial fix**
By putting `|y| + |ŷ|` in the denominator instead of just `y`, sMAPE softens (but doesn't fully eliminate) the near-zero blowup problem. It's "symmetric" because over- and under-predictions are penalized more evenly than in plain MAPE.

**MASE — the most important one conceptually**
This is the metric that directly encodes the golden rule from the Key Takeaways: *always compare against a naive baseline.* MASE = your model's MAE ÷ the naive model's MAE. A value:
- **< 1** → your model beats the naive "just predict yesterday's value" baseline
- **> 1** → your fancy model is actually *worse* than doing nothing clever at all — a strong signal something's wrong (overfitting, leakage, wrong model choice)
- **= 1** → your model performs exactly as well as the naive baseline (so why use it?)

### The code

```python
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Always compare against naive baseline!
naive_rmse = np.sqrt(mean_squared_error(test_ml, train_ml[-len(test_ml):].values))
model_rmse = np.sqrt(mean_squared_error(y_test_ml, y_pred_ml))
print(f"Naive: {naive_rmse:.2f}, Model: {model_rmse:.2f}")
print(f"Improvement: {(1 - model_rmse/naive_rmse)*100:.1f}%")
```

Two things worth pausing on:

1. **The naive baseline calculation** — `train_ml[-len(test_ml):].values` takes the *last N values of the training set* (where N = length of test set) as the "prediction" for the test set. This is a simple version of the naive forecast: literally saying "predict that the future will look like the most recent chunk of the past." (A more standard naive forecast is often "predict tomorrow = today," i.e., one-step lag — but this version is comparing whole blocks.)

2. **The improvement % formula** — `(1 - model_rmse/naive_rmse)*100` tells you what percentage better your model is than doing nothing sophisticated. If this comes out negative, your model is *worse* than the naive baseline — same red flag as MASE > 1.

### Pen & Paper Example: Differencing (tying back to Topic 3)

Input series (monthly): `[10, 12, 15, 13, 20, 22]` (index 0 to 5)

**First difference (d=1)**: `y_t - y_{t-1}`
- index 1: 12 − 10 = 2
- index 2: 15 − 12 = 3
- index 3: 13 − 15 = −2
- index 4: 20 − 13 = 7
- index 5: 22 − 20 = 2
- Result: `[NaN, 2, 3, -2, 7, 2]` — removes linear trend

**Seasonal difference (D=1, period=3)**: `y_t - y_{t-3}`
- index 3: 13 − 10 = 3
- index 4: 20 − 12 = 8
- index 5: 22 − 15 = 7
- Result: `[NaN, NaN, NaN, 3, 8, 7]` — removes periodic seasonality (period 3)

Try this yourself: compute the *seasonal difference with period=2* on the same series, by hand, before checking — it's good practice for internalizing the index arithmetic.

---

### Full picture — the Key Takeaways table, now that you've seen why each rule exists

| Rule | Why it matters |
|---|---|
| Never shuffle | Temporal ordering *is* the data structure |
| Stationarity first | ARIMA assumes constant statistics |
| Rebuild features after split | Prevents leakage in rolling stats |
| Use TimeSeriesSplit | Standard K-Fold leaks future into past |
| Compare vs naive | Model must beat "last value" to be useful |

That's the whole deck! You've now gone through: fundamentals → decomposition → stationarity → ACF/PACF → ARIMA/SARIMA → ML lag-feature forecasting → walk-forward validation → evaluation metrics — which is actually the complete real-world workflow in order.

Want me to quiz you on the whole thing to check retention, or work through a hands-on example end-to-end with a real (or synthetic) dataset?
