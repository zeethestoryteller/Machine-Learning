# Topic 5: ARIMA & SARIMA

### ARIMA parameters — what each letter means

| Parameter | Meaning |
|---|---|
| **p** | AR order: uses last `p` observations |
| **d** | Differencing order: how many times to difference |
| **q** | MA order: uses last `q` residuals (forecast errors) |

You've actually already learned how to figure out each of these:
- **d** comes from Topic 3 (Stationarity) — how many times you had to difference to pass the ADF test
- **p** and **q** come from Topic 4 (ACF/PACF) — read off the plots using the cutoff patterns

So ARIMA(p,d,q) isn't three arbitrary numbers — it's the direct output of the diagnostic work you just did.

### The ARIMA equation, concretely

```
ARIMA(1,1,1):
  1 AR term: ȳ_t = φ₁ȳ_{t-1} + ε_t + θ₁ε_{t-1}
  where ȳ_t = y_t - y_{t-1}  (first difference)
```

Breaking this down:
- The model doesn't operate on the raw `y_t` — it operates on the **differenced** series `ȳ_t` (since d=1).
- `φ₁ȳ_{t-1}` — the AR part: today's (differenced) value is partly explained by yesterday's (differenced) value, weighted by `φ₁`.
- `ε_t` — today's random shock/error.
- `θ₁ε_{t-1}` — the MA part: today's value is also partly explained by *yesterday's forecast error*, weighted by `θ₁`. This is the model "correcting" based on how wrong it was last time.

Once you have a forecast on `ȳ_t`, you integrate (cumulative sum) back up to get a forecast on the original `y_t` scale — that's literally what the "I" (Integrated) in ARIMA refers to.

### SARIMA — adding seasonality
Plain ARIMA has no way to represent "this month looks like the same month last year." SARIMA adds a second set of (P, D, Q, s) parameters that work exactly like (p, d, q) but at the **seasonal lag** `s` instead of lag 1.

```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ARIMA
model_arima = ARIMA(train, order=(1,1,1))
result_arima = model_arima.fit()
print(result_arima.summary())

# SARIMA (seasonal ARIMA)
# SARIMA(p,d,q)(P,D,Q,s) where s = seasonal period
model_sarima = SARIMAX(
    train,
    order=(1,1,1),                  # non-seasonal
    seasonal_order=(1,1,1,12),      # seasonal with period 12
    trend='n'
)
result_sarima = model_sarima.fit(disp=False)

# Forecast
forecast = result_sarima.get_forecast(steps=len(test))
pred     = forecast.predicted_mean
conf_int = forecast.conf_int(alpha=0.05)   # 95% CI
```

Key details:
- `SARIMAX` is used (not `SARIMA`) — the "X" stands for eXogenous variables support, but you don't have to use that feature; it works fine as plain SARIMA if you skip exogenous inputs.
- `seasonal_order=(1,1,1,12)` reads as: seasonal AR=1, seasonal differencing=1, seasonal MA=1, period=12 (monthly data, yearly cycle).
- `trend='n'` means no additional deterministic trend term is added on top of the differencing — since differencing (d=1) already handles the trend, adding a separate trend term would often be redundant/conflicting.
- `get_forecast()` (rather than just `.predict()`) is preferred because it gives you `conf_int()` — confidence intervals — which `.predict()` alone doesn't.

### Auto ARIMA — when you don't want to eyeball ACF/PACF manually

```python
# pip install pmdarima
from pmdarima import auto_arima

auto_model = auto_arima(
    train,
    seasonal=True, m=12,
    stepwise=True,              # efficient search
    suppress_warnings=True,
    information_criterion='aic',   # AIC, BIC, HQIC
    max_p=3, max_q=3, max_P=2, max_Q=2,
    d=None,                     # auto-determine d
    D=None                      # auto-determine D
)
print(auto_model.summary())
```

- `stepwise=True` uses a smart search (Hyndman-Khandakar algorithm) instead of brute-forcing every combination of p,d,q,P,D,Q — much faster.
- `d=None, D=None` lets the library run its own stationarity tests to pick differencing orders automatically, rather than you doing the ADF test manually.
- It picks the best model by **AIC** (Akaike Information Criterion) by default — lower AIC = better tradeoff between fit quality and model complexity (penalizes overfitting).
- **Practical tip**: auto_arima is great for a fast baseline, but it's worth understanding the manual ACF/PACF process (Topic 4) because auto-search can pick a locally-optimal but non-intuitive model, and you'll want to sanity-check its output.

### Quick check
If your ACF/PACF said p=1, q=1, and your ADF tests said you needed one regular difference and one seasonal difference (period=12) to become stationary — what would your `order` and `seasonal_order` arguments look like?

*(Answer: `order=(1,1,1)`, `seasonal_order=(P,1,Q,12)` — you'd still need ACF/PACF on the seasonally-differenced residuals to nail down P and Q specifically.)*

Ready for **Topic 6: ML-Based Forecasting with Lag Features**?
