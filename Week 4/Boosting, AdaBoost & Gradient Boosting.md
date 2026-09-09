# Boosting Estimators — Complete Guide
### AdaBoost & Gradient Boosting (Classifier + Regressor)

Boosting builds an ensemble **sequentially** — each new model tries to fix the mistakes of the previous ones — unlike bagging, where models are trained independently in parallel. This guide covers all four estimators from the slides: `AdaBoostClassifier`, `AdaBoostRegressor`, `GradientBoostingClassifier`, `GradientBoostingRegressor`.

---

## 1. How AdaBoost Works (Intuition)

1. Start by giving every training sample **equal weight**.
2. Train a weak learner (usually a shallow decision "stump").
3. Increase the weight of samples the model got **wrong**, decrease weight of samples it got **right**.
4. Train the next weak learner on this re-weighted data — it's forced to focus on the hard cases.
5. Repeat for `n_estimators` rounds.
6. Final prediction = weighted vote (classification) or weighted average (regression) of all weak learners, where each learner's vote is weighted by how accurate it was (`estimator_weights_`).

---

## 2. AdaBoostClassifier

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Data
X, y = make_classification(n_samples=1000, n_features=20,
                            n_informative=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 2. Model
clf = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # weak learner ("stump")
    n_estimators=50,
    learning_rate=1.0,
    algorithm="SAMME",     # newer sklearn versions default/only support SAMME
    random_state=42
)

# 3. Train
clf.fit(X_train, y_train)

# 4. Predict & evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
```

### Parameters — what they do and why they matter

| Parameter | Default | Why you tune it |
|---|---|---|
| `estimator` (older sklearn: `base_estimator`) | `DecisionTreeClassifier(max_depth=1)` | This is the "weak learner" boosted at every round. A depth-1 stump is deliberately weak — boosting works by combining many weak learners. If you make the base estimator too strong (deep tree), boosting adds little value and can overfit fast. |
| `n_estimators` | 50 | Number of boosting rounds / weak learners to combine. More rounds = more capacity to fit complex patterns, but too many can overfit, especially with `learning_rate` too high. This is the **stopping point** for boosting. |
| `learning_rate` | 1.0 | Shrinks the contribution of each classifier. A lower value means each weak learner has less say, so you typically need **more** `n_estimators` to compensate. This is the classic **bias-variance / n_estimators trade-off** mentioned in the slides — small learning rate + many estimators usually generalizes better than large learning rate + few estimators. |
| `algorithm` | `"SAMME"` | The boosting algorithm variant. `SAMME.R` (real-valued, probability-based) existed in older sklearn but was removed/deprecated in recent versions — `SAMME` is now standard. |
| `random_state` | `None` | Controls reproducibility of the (weighted) bootstrap resampling done internally. |

### Attributes (available after `.fit()`)

| Attribute | Meaning |
|---|---|
| `estimator_` (`base_estimator_`) | The base estimator template used. |
| `estimators_` | The actual list of fitted weak learners (one per round). |
| `estimator_weights_` | How much "vote" each weak learner gets in the final prediction — learners with lower error get higher weight. |
| `estimator_errors_` | The weighted classification error of each weak learner at the round it was trained — useful for diagnosing whether boosting is still improving or has plateaued. |

```python
print(clf.estimator_weights_)   # importance of each stump
print(clf.estimator_errors_)    # error of each stump
```

---

## 3. AdaBoostRegressor

```python
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. Data
X, y = make_regression(n_samples=1000, n_features=20,
                        noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 2. Model
reg = AdaBoostRegressor(
    estimator=DecisionTreeRegressor(max_depth=3),   # default depth is 3 for regressor
    n_estimators=50,
    learning_rate=1.0,
    loss="linear",          # regressor-specific parameter
    random_state=42
)

# 3. Train
reg.fit(X_train, y_train)

# 4. Predict & evaluate
y_pred = reg.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
```

### Parameters — what they do and why they matter

| Parameter | Default | Why you tune it |
|---|---|---|
| `estimator` (`base_estimator`) | `DecisionTreeRegressor(max_depth=3)` | Note the default depth here is **3**, deeper than the classifier's default of 1 — a regression stump of depth 1 is often too weak to capture any real trend, so sklearn allows a bit more capacity by default. |
| `n_estimators` | 50 | Same idea as classifier — max number of boosting rounds before boosting is terminated. |
| `learning_rate` | 1.0 | Shrinks each regressor's contribution to the final weighted average prediction. Same trade-off with `n_estimators` as in the classifier. |
| `loss` | `"linear"` | Regressor-only parameter — controls how per-sample error is converted into new sample weights each round. Options: `"linear"` (proportional to error), `"square"` (squares the error, penalizing large errors more), `"exponential"` (very aggressively penalizes large errors). Choose `"square"`/`"exponential"` if you want the model to focus harder on outlier-like hard examples. |
| `random_state` | `None` | Reproducibility. |

### Attributes

Same four as the classifier: `estimator_`, `estimators_`, `estimator_weights_`, `estimator_errors_` — interpreted the same way, but errors here are weighted regression losses instead of classification error rates.

---

## 4. Tuning Tips for AdaBoost (from the slides)

> "The main parameters to tune to obtain good results are **n_estimators** and the complexity of the base estimators (e.g., its depth `max_depth` or `min_samples_split`)."

- Increase `n_estimators` → more rounds of correction, but watch for overfitting on noisy data (AdaBoost is sensitive to outliers/noise since it keeps up-weighting hard/mislabeled points).
- Increase base estimator's `max_depth` → each weak learner becomes stronger, needing fewer boosting rounds — but this moves away from "weak learner" boosting philosophy and risks overfitting.
- Lower `learning_rate` + higher `n_estimators` generally gives smoother, more generalizable ensembles at the cost of training time.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.1, 1.0],
    "estimator__max_depth": [1, 2, 3]
}

grid = GridSearchCV(
    AdaBoostClassifier(estimator=DecisionTreeClassifier()),
    param_grid, cv=5, scoring="accuracy"
)
grid.fit(X_train, y_train)
print(grid.best_params_)
```

---

## 5. Gradient Boosting — How It's Different from AdaBoost

Instead of re-weighting misclassified samples, Gradient Boosting fits each new tree to the **residual errors** (or, more precisely, the negative gradient of the loss function) of the current ensemble. This makes it a general framework: any differentiable loss function can be optimized.

---

## 6. GradientBoostingClassifier

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. Data
X, y = make_classification(n_samples=1000, n_features=20,
                            n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 2. Model
gbc = GradientBoostingClassifier(
    loss="log_loss",          # binary/multiclass classification loss
    learning_rate=0.1,
    n_estimators=100,
    subsample=1.0,
    criterion="friedman_mse",
    min_samples_split=2,
    min_samples_leaf=1,
    max_depth=3,
    max_features=None,
    random_state=42
)

# 3. Train
gbc.fit(X_train, y_train)

# 4. Predict & evaluate
y_pred = gbc.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

### Parameters — what they do and why they matter

| Parameter | Default | Why you tune it |
|---|---|---|
| `loss` | `"log_loss"` | The function being minimized. `"log_loss"` = logistic loss, works for binary **and** multiclass (as noted in the slides: "supports both binary and multiclass classification"). |
| `learning_rate` | 0.1 | Shrinks each tree's contribution to the running prediction. The **most important parameter alongside `n_estimators`** per the slides — smaller values need more trees but usually generalize better ("shrinkage"). |
| `n_estimators` | 100 | Number of boosting stages (trees) to fit. Too many can overfit if `learning_rate` isn't small enough — always tune these two together. |
| `subsample` | 1.0 | Fraction of training samples used to fit each tree. Values < 1.0 introduce randomness (stochastic gradient boosting), which can reduce variance/overfitting, similar in spirit to bagging. |
| `criterion` | `"friedman_mse"` | The function used to measure split quality inside each tree — Friedman's improved version of MSE, tailored for boosting. |
| `min_samples_split` | 2 | Minimum samples required to split an internal node — raising this **limits tree complexity**, reducing overfitting. |
| `min_samples_leaf` | 1 | Minimum samples required in a leaf node — same purpose: higher value = simpler, smoother trees. |
| `max_depth` | 3 | Controls how complex each individual tree can be. Since boosting trees are meant to be relatively weak, this is usually kept shallow (3–5). |
| `max_features` | `None` (all features) | Number/fraction of features considered at each split — restricting this adds randomness and can reduce overfitting/speed up training. |
| `random_state` | `None` | Reproducibility of subsampling and any random tie-breaking. |

### Attributes

```python
print(gbc.feature_importances_)   # which features mattered most
print(gbc.train_score_)           # training loss at each boosting stage
```

- `feature_importances_` — relative importance of each input feature across all trees.
- `train_score_` — loss value at each stage, useful for plotting a learning curve to pick a good `n_estimators`.
- `estimators_` — the underlying array of fitted regression trees (classification is done via multiple underlying regression trees + a link function).

---

## 7. GradientBoostingRegressor

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Data
X, y = make_regression(n_samples=1000, n_features=20,
                        noise=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 2. Model
gbr = GradientBoostingRegressor(
    loss="squared_error",     # regression loss function
    learning_rate=0.1,
    n_estimators=100,
    subsample=1.0,
    criterion="friedman_mse",
    min_samples_split=2,
    min_samples_leaf=1,
    max_depth=3,
    max_features=None,
    random_state=42
)

# 3. Train
gbr.fit(X_train, y_train)

# 4. Predict & evaluate
y_pred = gbr.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))
```

### Parameters — what they do and why they matter

Same core set as the classifier, with one key difference:

| Parameter | Default | Why you tune it |
|---|---|---|
| `loss` | `"squared_error"` | Regression-specific loss. Options include `"squared_error"` (standard, sensitive to outliers), `"absolute_error"` (robust to outliers), `"huber"` (blend of both — robust but still smooth), `"quantile"` (predict a specific percentile, useful for prediction intervals). |
| `alpha` | 0.9 | Only used with `loss="huber"` or `"quantile"` — the quantile/threshold value. |
| `learning_rate`, `n_estimators`, `subsample`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `random_state` | (same as classifier above) | Same meaning and same tuning trade-offs as `GradientBoostingClassifier`. |

### Attributes

Same as classifier: `feature_importances_`, `train_score_`, `estimators_`.

---

## 8. Choosing n_estimators & learning_rate Together (Practical Recipe)

```python
import matplotlib.pyplot as plt

gbr = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05,
                                 max_depth=3, random_state=42)
gbr.fit(X_train, y_train)

# staged_predict lets you see error at every boosting stage
test_errors = [
    mean_squared_error(y_test, y_pred_stage)
    for y_pred_stage in gbr.staged_predict(X_test)
]

plt.plot(test_errors)
plt.xlabel("Boosting iteration")
plt.ylabel("Test MSE")
plt.title("Finding the sweet spot for n_estimators")
plt.show()

best_n_estimators = int(np.argmin(test_errors)) + 1
print("Best n_estimators:", best_n_estimators)
```

This is the standard way to pick `n_estimators` without a full grid search: train with a large number, then look at `staged_predict` (or `staged_decision_function` for classifiers) to see exactly where test error stops improving — that's your best stopping point, and it directly demonstrates the `n_estimators` ↔ `learning_rate` trade-off mentioned throughout the slides.

---

## 9. Quick Summary Table

| Estimator | Base learner default | Key extra parameter | Handles multiclass? |
|---|---|---|---|
| `AdaBoostClassifier` | Stump, `max_depth=1` | `algorithm` | Yes (via SAMME) |
| `AdaBoostRegressor` | Tree, `max_depth=3` | `loss` (`linear`/`square`/`exponential`) | N/A |
| `GradientBoostingClassifier` | Tree, `max_depth=3` | `loss` (`log_loss`) | Yes |
| `GradientBoostingRegressor` | Tree, `max_depth=3` | `loss` (`squared_error`/`huber`/`quantile`) | N/A |

**Note:** The slides mention XGBoost is demonstrated separately via Colab — XGBoost is a separate, more optimized/regularized implementation of gradient boosting (not part of `sklearn.ensemble`), with its own library (`xgboost`) and additional parameters like `reg_lambda`, `reg_alpha`, and built-in handling of missing values.
