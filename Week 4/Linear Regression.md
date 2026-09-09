# Linear Regression — Complete Implementation Guide
*(Based on IIT Madras ML Practice, Week 4)*

---

## 0. The Big Picture

Linear regression predicts a target as a **weighted sum of features**:

```
ŷ = w0 + w1*x1 + w2*x2 + ... + wm*xm  =  wᵀx
```

- `w0` is the **intercept** (bias)
- `w1...wm` are the **coefficients** (weights)

sklearn gives you **two ways** to find these weights:

| Approach | Class | How it solves for w |
|---|---|---|
| Normal equation (closed-form) | `LinearRegression` | Exact linear algebra solution |
| Iterative optimization | `SGDRegressor` | Gradient descent, step by step |

**Rule of thumb:** use `LinearRegression` for small/medium data. Switch to `SGDRegressor` when you have **>10,000 samples**, since computing the exact solution becomes expensive.

Both are **estimators** — meaning they follow the same 3-step sklearn pattern:
```python
model = SomeEstimator(...)   # 1. instantiate
model.fit(X_train, y_train)  # 2. train
model.predict(X_test)        # 3. predict
```
This same pattern works for every regression estimator you'll ever use in sklearn — that consistency is the whole point of the "estimator" interface.

---

## 1. Baseline Model First (Always Do This)

Before building a real model, build a "dumb" baseline to know what score you need to beat.

```python
from sklearn.dummy import DummyRegressor

dummy_regr = DummyRegressor(strategy="mean")
dummy_regr.fit(X_train, y_train)
dummy_regr.predict(X_test)
dummy_regr.score(X_test, y_test)
```

**`strategy` options:**
| Strategy | Prediction made |
|---|---|
| `"mean"` | Always predicts the mean of `y_train` |
| `"median"` | Always predicts the median of `y_train` |
| `"quantile"` | Predicts a specific quantile (you also set `quantile=0.x`) |
| `"constant"` | Predicts a value you specify (you also set `constant=value`) |

If your real model can't beat this, something is wrong.

---

## 2. Training a Linear Regression Model

### Step 1 — Instantiate

```python
# Option A: Normal equation (exact solution)
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()

# Option B: Iterative optimization (for large data)
from sklearn.linear_model import SGDRegressor
linear_regressor = SGDRegressor()
```

### Step 2 — Fit

```python
linear_regressor.fit(X_train, y_train)
```

This works for **both single-output and multi-output** regression (i.e., `y_train` can be a vector or a matrix of multiple targets).

---

## 3. SGDRegressor — Every Parameter Explained

`SGDRegressor` implements **Stochastic Gradient Descent**. It's the "manual transmission" version of linear regression — you get much more control, but you have to understand the knobs.

### 3.1 `random_state` — Reproducibility
```python
SGDRegressor(random_state=42)
```
SGD has randomness in it (shuffling, initialization). Always set a seed so your results are reproducible. **Set this every time you code**, even though the slides skip it for brevity.

### 3.2 Feature Scaling (not a parameter, but mandatory prep)
SGD is **very sensitive** to feature scale. Always scale your features first, using a `Pipeline`:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sgd = Pipeline([
    ('feature_scaling', StandardScaler()),
    ('sgd_regressor', SGDRegressor())
])
sgd.fit(X_train, y_train)
```
**Exceptions** — you don't need to scale:
- Word frequency / count features (already have intrinsic scale)
- Indicator (0/1) features

For PCA-extracted features: scale by a constant `c` so the **average L2 norm of training data = 1**.

### 3.3 `shuffle` — Reshuffle data each epoch
```python
SGDRegressor(shuffle=True)   # default is True
```
Shuffles training data after every full pass (epoch), which helps SGD converge better and avoid cyclic patterns.

### 3.4 `learning_rate` and `eta0` — Step size control

| `learning_rate` value | Behavior |
|---|---|
| `'constant'` | Learning rate = `eta0` throughout training |
| `'invscaling'` (**default**) | Shrinks every iteration: `eta = eta0 / pow(t, power_t)` |
| `'adaptive'` | Stays at `eta0` while loss improves; divided by 5 when stuck; **stops training entirely once learning rate < 1e-6** |

**Default settings:** `learning_rate='invscaling'`, `eta0=0.01`, `power_t=0.25`

```python
# Constant learning rate
SGDRegressor(learning_rate='constant', eta0=1e-2)

# Adaptive learning rate
SGDRegressor(learning_rate='adaptive', eta0=1e-2)
```
Tune `eta0` and `power_t` to speed up or slow down convergence.

### 3.5 `max_iter` — Number of epochs
```python
SGDRegressor(max_iter=100)   # default = 1000
```
One **epoch** = one full pass over the entire training set.

**Practical tip for picking `max_iter`:** SGD tends to converge after seeing roughly **10⁶ total training samples**. So for `n` samples:
```python
max_iter = np.ceil(10**6 / n)
```

### 3.6 Stopping Criteria — Two Options

**Option 1: Based on training loss** (`tol`, `n_iter_no_change`, `max_iter`)
```python
SGDRegressor(loss='squared_error', max_iter=500,
             tol=1e-3, n_iter_no_change=5)
```
Stops when training loss fails to improve by more than `tol` for `n_iter_no_change` consecutive epochs — otherwise stops at `max_iter`.

**Option 2: Based on a held-out validation set** (`early_stopping`, `validation_fraction`)
```python
SGDRegressor(loss='squared_error', early_stopping=True,
             max_iter=500, tol=1e-3,
             validation_fraction=0.2, n_iter_no_change=5)
```
Sets aside `validation_fraction` of the training data as a validation set (scored via `.score()`), and stops when the **validation score** doesn't improve by `tol` for `n_iter_no_change` epochs. This is generally the more reliable option since it checks generalization, not just training fit.

### 3.7 `loss` — The loss function
```python
SGDRegressor(loss='squared_error')   # this course focuses on this one
```
sklearn also supports `'huber'` and others (see sklearn docs) — useful for outlier-robust regression, but `'squared_error'` is the standard choice for plain linear regression.

### 3.8 `penalty` — Regularization type
```python
SGDRegressor(penalty='l2')   # options: 'l1', 'l2', 'elasticnet', or None
```
This controls what kind of regularization is added (Ridge-style `l2`, Lasso-style `l1`, or a mix `elasticnet`). You'll likely cover this in more depth in a regularization module.

### 3.9 `average` — Averaged SGD
Averaged SGD keeps a running **average of past weight vectors** instead of just using the latest one — this reduces variance in noisy SGD updates.

```python
# Option 1: average across ALL updates from the start
SGDRegressor(average=True)

# Option 2: only start averaging after seeing N samples
SGDRegressor(average=10)   # begins averaging after 10 samples seen
```
Averaged SGD tends to work best with a **larger number of features** and a **higher `eta0`**.

### 3.10 `warm_start` — Continue training instead of restarting
```python
SGDRegressor(warm_start=True)   # default: False
```
When `True`, calling `.fit()` again **continues from the previous weights** instead of reinitializing. This is the trick used to manually monitor loss epoch-by-epoch:

```python
sgd_reg = SGDRegressor(max_iter=1, tol=-np.infty, warm_start=True,
                        penalty=None, learning_rate="constant", eta0=0.0005)

for epoch in range(1000):
    sgd_reg.fit(X_train, y_train)              # continues where it left off
    y_val_predict = sgd_reg.predict(X_val)
    val_error = mean_squared_error(y_val, y_val_predict)
    # -> log/plot val_error to watch training progress
```

### 3.11 Full Parameter Cheat Sheet

```python
from sklearn.linear_model import SGDRegressor

model = SGDRegressor(
    loss='squared_error',       # loss function
    penalty='l2',                # regularization: 'l1','l2','elasticnet', None
    max_iter=1000,                # number of epochs
    tol=1e-3,                     # stopping tolerance
    n_iter_no_change=5,           # epochs to wait before stopping
    early_stopping=False,         # use validation-based stopping?
    validation_fraction=0.1,      # fraction held out if early_stopping=True
    learning_rate='invscaling',   # 'constant','invscaling','adaptive'
    eta0=0.01,                    # initial learning rate
    power_t=0.25,                 # invscaling decay exponent
    shuffle=True,                 # reshuffle data each epoch
    average=False,                # averaged SGD (True or int)
    warm_start=False,             # continue from previous fit?
    random_state=42               # reproducibility
)
```

---

## 4. Model Inspection — Reading the Learned Weights

Works identically for `LinearRegression`, `SGDRegressor`, and every other regression estimator (because they're all "estimators" following the same API):

```python
linear_regressor.coef_       # weights w1, w2, ..., wm
linear_regressor.intercept_  # bias w0
```

---

## 5. Model Inference — Making Predictions

```python
# X_test shape: (#samples, #features)
linear_regressor.predict(X_test)
```
Same code works across **all** regression estimators.

---

## 6. Model Evaluation

### 6.1 General Evaluation Workflow
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
# 1. Split data (above)
# 2. Fit on X_train, y_train
# 3. Compute training error (empirical error)
# 4. Compute test error (generalization error)
# 5. Compare the two — large gap = overfitting
```

### 6.2 `.score()` — R² / Coefficient of Determination

```python
linear_regressor.score(X_test, y_test)
```

Formula:
```
R² = 1 − (u / v)

u = residual sum of squares = (Xw − y)ᵀ(Xw − y)     [error of YOUR model]
v = total sum of squares    = (y − ȳ)ᵀ(y − ȳ)        [error of a "predict-the-mean" model]
```

**How to interpret it:**
- **R² = 1.0** → perfect predictions (u = 0)
- **R² = 0.0** → your model is exactly as good as always predicting the mean (u = v)
- **R² < 0** → your model is *worse* than just predicting the mean — this is possible! (model can be arbitrarily bad)

### 6.3 Explicit Evaluation Metrics

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mean_absolute_error(y_test, y_predicted)
mean_squared_error(y_test, y_predicted)
r2_score(y_test, y_predicted)          # same value as .score()
```
All of these also work for **multi-output** regression.

**More specialized metrics:**

| Metric | Import | Best used when... |
|---|---|---|
| `mean_squared_log_error` | `sklearn.metrics` | Target grows exponentially (population, sales); penalizes **under-estimation** more heavily |
| `mean_absolute_percentage_error` | `sklearn.metrics` | You care about **relative** error |
| `median_absolute_error` | `sklearn.metrics` | You want a metric **robust to outliers** |
| `max_error` | `sklearn.metrics` | You want the **worst-case** error — ⚠️ single-output only, no multi-output support |

```python
from sklearn.metrics import max_error
train_error = max_error(y_train, y_predicted)
test_error  = max_error(y_test, y_predicted)
```

### 6.4 Scores vs Errors — the `neg_` naming convention

- **Score** = higher is better
- **Error** = lower is better

sklearn's cross-validation tools (below) always expect "higher is better," so error metrics get a `neg_` prefix when used as a `scoring` string:

| Metric function | Scoring string |
|---|---|
| `mean_absolute_error` | `'neg_mean_absolute_error'` |
| `mean_squared_error` | `'neg_mean_squared_error'` |
| `mean_squared_error` (rooted) | `'neg_root_mean_squared_error'` |
| `mean_squared_log_error` | `'neg_mean_squared_log_error'` |
| `median_absolute_error` | `'neg_median_absolute_error'` |
| (R²) | `'r2'` |
| (worst case) | `'max_error'` |

---

## 7. Cross-Validation — Robust Evaluation

**Why not trust a single train/test split?**
1. If the test set is small, the test error is unstable and unreliable.
2. By chance, the "easy" examples might have landed in the test set, giving an overly optimistic error estimate.

**Solution:** cross-validation — repeatedly split the data, train and evaluate many times, and look at the *distribution* of errors, not just one number.

### 7.1 Cross-Validation Iterators

| Iterator | What it does |
|---|---|
| `KFold` | Splits data into k folds; each fold takes a turn as the test set, rest are training |
| `RepeatedKFold` | Repeats KFold multiple times with different randomization |
| `LeaveOneOut` | Extreme case of KFold where k = n (each single sample is its own test fold) |
| `ShuffleSplit` | Randomly shuffles + splits into train/test, `n_splits` times (splits can overlap) |

### 7.2 `cross_val_score` — Quick Scoring

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
score = cross_val_score(lin_reg, X, y, cv=5)   # 5-fold CV, uses KFold internally
```

Equivalent explicit version:
```python
from sklearn.model_selection import KFold

kfold_cv = KFold(n_splits=5, random_state=42, shuffle=True)  # shuffle needed to use random_state
score = cross_val_score(lin_reg, X, y, cv=kfold_cv)
```

**Using `LeaveOneOut`:**
```python
from sklearn.model_selection import LeaveOneOut

loocv = LeaveOneOut()
score = cross_val_score(lin_reg, X, y, cv=loocv)
# equivalent to KFold(n_splits=n) where n = X.shape[0]
```

**Using `ShuffleSplit`:**
```python
from sklearn.model_selection import ShuffleSplit

shuffle_split = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
score = cross_val_score(lin_reg, X, y, cv=shuffle_split)
```

**Specifying a metric** (default is R² if you don't specify):
```python
score = cross_val_score(lin_reg, X, y, cv=shuffle_split,
                         scoring='neg_mean_absolute_error')
```

### 7.3 `cross_validate` — More Detail Than `cross_val_score`

```python
from sklearn.model_selection import cross_validate, ShuffleSplit

cv = ShuffleSplit(n_splits=40, test_size=0.3, random_state=0)
cv_results = cross_validate(
    regressor, data, target, cv=cv, scoring="neg_mean_absolute_error")
```

Returns a **dictionary** with keys:
- `fit_time`
- `score_time`
- `test_score`
- `estimator` — only if `return_estimator=True`
- `train_score` — only if `return_train_score=True`

**Getting trained models + training scores back:**
```python
cv_results = cross_validate(
    regressor, data, target,
    cv=cv, scoring="neg_mean_absolute_error",
    return_train_score=True,
    return_estimator=True)

cv_results['estimator']   # list of trained models, one per fold
```

**Multiple metrics at once** (this is the key advantage over `cross_val_score`, which only supports one metric):
```python
cv_results = cross_validate(
    regressor, data, target,
    cv=cv,
    scoring=["neg_mean_absolute_error", "neg_mean_squared_error"],
    return_train_score=True,
    return_estimator=True)
```

---

## 8. Diagnosing Underfitting / Overfitting

### 8.1 Learning Curves (effect of #samples)

```python
from sklearn.model_selection import learning_curve

results = learning_curve(
    lin_reg, X_train, y_train, train_sizes=train_sizes, cv=cv,
    scoring="neg_mean_absolute_error")

train_size, train_scores, test_scores = results[:3]
train_errors, test_errors = -train_scores, -test_scores   # convert scores back to errors
```
Plot `train_errors` and `test_errors` against `train_size` to see how error evolves as you feed the model more data.

### 8.2 General Under/Overfitting Diagnosis (any hyperparameter)

1. Fit multiple models, varying **one hyperparameter** at a time (e.g., number of features, or `max_iter`, or `penalty` strength).
2. For each model, compute training error and test error.
3. Plot **hyperparameter value vs. error** (two lines: train and test).
4. Read the plot:
   - **Both errors high** → underfitting → model too simple / needs more capacity
   - **Train error low, test error high (big gap)** → overfitting → model too complex / needs regularization or more data
   - **Both errors low and close together** → good fit

This same 4-step recipe generalizes to tuning **any** hyperparameter, not just #features.

---

## Quick-Reference: Full Workflow Template

```python
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, ShuffleSplit
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 2. Baseline
dummy = DummyRegressor(strategy="mean")
dummy.fit(X_train, y_train)
print("Baseline R2:", dummy.score(X_test, y_test))

# 3. Real model (choose based on data size)
if X_train.shape[0] > 10000:
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('sgd', SGDRegressor(loss='squared_error', penalty='l2',
                              learning_rate='invscaling', eta0=0.01,
                              max_iter=int(np.ceil(1e6 / X_train.shape[0])),
                              tol=1e-3, n_iter_no_change=5,
                              random_state=42))
    ])
else:
    model = LinearRegression()

# 4. Fit
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
print("Test R2:", r2_score(y_test, y_pred))
print("Test MAE:", mean_absolute_error(y_test, y_pred))
print("Test RMSE:", mean_squared_error(y_test, y_pred, squared=False))

# 6. Cross-validate for robustness
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
cv_results = cross_validate(model, X, y, cv=cv,
                             scoring=["neg_mean_absolute_error", "r2"],
                             return_train_score=True)
print("CV test MAE (mean):", -cv_results['test_neg_mean_absolute_error'].mean())
```
