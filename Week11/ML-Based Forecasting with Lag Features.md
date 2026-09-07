## Topic 6: ML-Based Forecasting with Lag Features

### The core idea — reframing the problem
ARIMA/SARIMA are statistical models with built-in temporal structure. But you can also treat forecasting as a **plain supervised learning problem**: turn "predict the next value" into "predict `y` from a set of engineered features `X`," then throw any regressor (gradient boosting, random forest, etc.) at it.

The trick is entirely in how you build `X` — you have to manufacture features that encode "what happened in the past" since a generic regressor has no built-in sense of time order.

### The code

```python
# Convert time series to supervised learning problem
def create_lag_features(series, lags=[1,7,14,30], rolls=[7,14,30]):
    df = series.to_frame()
    col = series.name

    # Lag features
    for lag in lags:
        df[f'lag_{lag}'] = df[col].shift(lag)

    # Rolling statistics
    for window in rolls:
        df[f'roll_mean_{window}'] = df[col].shift(1).rolling(window).mean()
        df[f'roll_std_{window}']  = df[col].shift(1).rolling(window).std()
        df[f'roll_min_{window}']  = df[col].shift(1).rolling(window).min()
        df[f'roll_max_{window}']  = df[col].shift(1).rolling(window).max()

    # Calendar features
    df['year']       = df.index.year
    df['month']      = df.index.month
    df['dayofweek']  = df.index.dayofweek
    df['quarter']    = df.index.quarter
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

    return df.dropna()
```

Let's unpack each feature group and *why* it's there:

**1. Lag features** (`shift(lag)`)
`df[col].shift(1)` moves every value forward by one row, so on row `t` you now have `y_{t-1}` sitting as a feature. This directly gives the model "yesterday's value," "a week ago's value," etc. — analogous to what the AR term does in ARIMA, but explicit and usable by any ML model.

**2. Rolling statistics** — note the critical detail: `.shift(1).rolling(window).mean()`
The `.shift(1)` **before** `.rolling()` is not optional decoration — it's the single most important line to understand in this whole function. Without it, `rolling(7).mean()` computed at row `t` would include `y_t` itself (today's actual value) in the average used to predict `y_t` — that's **data leakage**. By shifting first, the rolling window only ever sees data up through `y_{t-1}`, so the feature is genuinely available at prediction time.

**3. Calendar features**
Things like month, day-of-week, quarter, is_weekend give the model explicit access to seasonal/cyclical signals it otherwise has no way to detect (a boosted tree doesn't know "this row is in December" unless you tell it).

**4. `dropna()`** at the end
Because of all the shifting/rolling, the earliest rows will have `NaN`s (e.g., you can't compute a `lag_30` for the 5th row of your data). These rows get dropped — you lose the first `max(lags/rolls)` rows of your dataset.

### Splitting and fitting

```python
df_ml = create_lag_features(ts['value'])

# Time-series split (no shuffle!)
split_date = '2022-01-01'
train_ml = df_ml[df_ml.index < split_date]
test_ml  = df_ml[df_ml.index >= split_date]

X_train_ml = train_ml.drop(columns=['value'])
y_train_ml = train_ml['value']
X_test_ml  = test_ml.drop(columns=['value'])
y_test_ml  = test_ml['value']

from sklearn.ensemble import GradientBoostingRegressor
gb_ts = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05)
gb_ts.fit(X_train_ml, y_train_ml)
y_pred_ml = gb_ts.predict(X_test_ml)
```

Notice: **the split is done by date, not `train_test_split()` with shuffling.** This is the same "never shuffle" rule from Topic 1 — a random split would let the model train on rows *after* the test period and peek at the future, inflating your accuracy dishonestly.

### An important subtlety worth knowing (not explicit in the PDF, but matters in practice)
This single-shot approach predicts one step ahead using true lag values from the training data. But when you want to forecast *multiple steps into the future* (not just the one row right after training ends), you don't have real `y_{t-1}` values anymore for `t` further out — you'd need to feed the model's own previous predictions back in as lag features recursively. That's a more advanced pattern than what's shown here, but good to be aware it's coming up if you build a multi-step forecaster.

### Quick check
Why would forgetting the `.shift(1)` before `.rolling(window).mean()` make your model look great during testing but fail in production?

*(Answer: During evaluation, if `y_t` leaks into `roll_mean` used to predict `y_t`, the model is essentially "cheating" — it looks accurate on historical test data but in true production, you don't have `y_t` yet when you need to predict it.)*

Ready for **Topic 7: Walk-Forward Validation**?
