# Topic 7: Walk-Forward Validation

### The problem it solves
Standard K-Fold cross-validation splits data into random folds and validates across all of them. For time series, this is fundamentally broken — a random fold could put "future" data in your training set and "past" data in your test set, which means your model gets to train on information that (chronologically) hadn't happened yet at prediction time. Your validation scores would look great and be completely dishonest.

**Walk-forward validation** fixes this by always keeping training data strictly *before* test data in time — and it repeats this process over multiple expanding windows, so you get several honest out-of-sample estimates instead of just one train/test split.

### The code

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5, gap=0)

scores = []
for fold, (train_idx, test_idx) in enumerate(tscv.split(X_ml)):
    X_tr, X_te = X_ml.iloc[train_idx], X_ml.iloc[test_idx]
    y_tr, y_te = y_ml.iloc[train_idx], y_ml.iloc[test_idx]

    model = GradientBoostingRegressor(n_estimators=100)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    rmse = np.sqrt(mean_squared_error(y_te, pred))
    scores.append(rmse)
    print(f"Fold {fold+1} RMSE: {rmse:.2f}")

print(f"Mean RMSE: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
```

### How `TimeSeriesSplit` actually divides the data
This is the part worth visualizing mentally. With `n_splits=5`, on a dataset ordered in time, it produces folds like this (schematically):

```
Fold 1: train=[1..............], test=[........]
Fold 2: train=[1..................], test=[........]
Fold 3: train=[1......................], test=[........]
Fold 4: train=[1..........................], test=[........]
Fold 5: train=[1..............................], test=[........]
```

Each fold's **training set expands** to include everything up to that point, and the **test set is always the chunk immediately after it** — never before, never overlapping with training. This is sometimes called an "expanding window" scheme. (There's also a "sliding window" variant where the training window has a fixed size and slides forward rather than growing, but `TimeSeriesSplit` by default expands.)

### `gap=0` parameter
`gap` lets you insert a buffer between the end of the training set and the start of the test set. Why would you ever want a gap? If your features involve rolling windows or lags, sometimes there's subtle leakage right at the boundary, or in real deployment there's a delay between "data available" and "prediction needed" (e.g., you always find out yesterday's true value with a 2-day reporting lag). `gap=0` here means no buffer — test starts immediately after train ends.

### Why report mean ± std, not just mean
`np.mean(scores)` alone hides how *consistent* the model is across time periods. `np.std(scores)` tells you: does this model perform reliably across different time windows, or does it do great in some periods and terrible in others (e.g., it might completely fall apart during periods with regime changes, holidays, or anomalies)? A model with a low mean RMSE but high std is a much riskier model to deploy than one with a slightly higher mean but low variance across folds.

### Quick check
If Fold 1 has 100 training rows and Fold 5 has 500 training rows, is the test set size also growing across folds, or does it stay the same?

*(Answer: With default settings, `TimeSeriesSplit` keeps test set sizes roughly equal across folds — it's the training set that expands each time, not the test size.)*

Ready for **Topic 8: Evaluation Metrics** — the final one?
