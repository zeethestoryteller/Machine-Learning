# Regularization in sklearn — Complete Guide

Regularization adds a penalty term to the loss function to shrink model weights and reduce overfitting. Here's everything from your slides, organized so you can actually implement it.

## The Core Idea

```
Regularized Loss = Sum of Squared Error + regularization_rate × penalty
```

The `alpha` parameter controls **regularization_rate** — how strongly you penalize large weights. Higher `alpha` = stronger regularization = simpler model.

---

## 1. Ridge Regression (L2 penalty)

Ridge shrinks weights smoothly but rarely makes them exactly zero.

**Option 1: Dedicated `Ridge` estimator**
```python
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1e-3)
```
- `alpha`: regularization rate (required)
- Works with `fit`, `score`, `predict` just like any linear regression

**Option 2: `SGDRegressor` with L2 penalty**
```python
from sklearn.linear_model import SGDRegressor
sgd = SGDRegressor(alpha=1e-3, penalty='l2')
```
- `penalty='l2'` is actually the **default** in SGDRegressor

---

## 2. Lasso Regression (L1 penalty)

Lasso can shrink weights all the way to zero → gives **sparse solutions** (automatic feature selection).

**Option 1: Dedicated `Lasso` estimator**
```python
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=1e-3)
```

**Option 2: `SGDRegressor` with L1 penalty**
```python
sgd = SGDRegressor(alpha=1e-3, penalty='l1')
```

sklearn also has `LassoLars`, which uses the least-angle-regression algorithm instead of coordinate descent — an alternative solver for the same L1-penalized problem.

---

## 3. ElasticNet (L1 + L2 combined)

Convex combination of both penalties:
```
penalty = (1 - l1_ratio) * L2 + l1_ratio * L1
```

```python
from sklearn.linear_model import SGDRegressor
sgd = SGDRegressor(penalty='elasticnet', l1_ratio=0.3, alpha=1e-3)
```
- `l1_ratio=0.3` → 30% L1, 70% L2 (L2 dominates here)
- `l1_ratio=1.0` → pure Lasso
- `l1_ratio=0.0` → pure Ridge

---

## 4. Regularization in a Polynomial Regression Pipeline

Since polynomial regression = polynomial transform + linear model, just swap in a regularized regressor:

```python
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

poly_model = Pipeline([
    ('polynomial_transform', PolynomialFeatures(degree=2)),
    ('ridge', Ridge(alpha=1e-3))
])
poly_model.fit(X_train, y_train)
```

Same pattern works for `Lasso` or `SGDRegressor(penalty='elasticnet', l1_ratio=0.3)`.

---

## 5. Finding the Best `alpha` Automatically

Instead of manually tuning `alpha`, sklearn provides model-specific CV estimators that search efficiently:

| Estimator | Purpose |
|---|---|
| `linear_model.RidgeCV` | Ridge + built-in CV for `alpha` |
| `linear_model.LassoCV` | Lasso + built-in CV for `alpha` |
| `linear_model.ElasticNetCV` | ElasticNet + built-in CV |

These are efficient because certain models can fit a *range* of alpha values almost as cheaply as fitting one value.

**`RidgeCV` parameters:**
- `alphas`: list of regularization rates to try (must be positive; larger = stronger regularization)
- `cv`: cross-validation strategy
  - `None` → efficient Leave-One-Out CV
  - integer → number of folds
  - a CV splitter object
  - an iterable of (train, test) index arrays

  For binary/multiclass problems with `cv=None` or an integer, sklearn uses `StratifiedKFold`; otherwise `KFold`.

**Model inspection with RidgeCV:**
- `coef_`, `intercept_` — same as usual
- `alphas` — extra output showing the estimated best regularization parameter

```python
from sklearn.linear_model import RidgeCV
ridge_cv = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10], cv=5)
ridge_cv.fit(X_train, y_train)
print(ridge_cv.alpha_)  # best alpha found
```

**Alternative:** manually cross-validate `Ridge`/`Lasso`/`SGDRegressor` with `GridSearchCV` or `RandomizedSearchCV` (covered in your HPT slides) if you want more control (e.g., tuning `alpha` jointly with polynomial `degree`).

---

## Quick Decision Guide

| Situation | Use |
|---|---|
| Want to shrink weights, keep all features | Ridge (L2) |
| Want automatic feature selection / sparsity | Lasso (L1) |
| Want a blend of both | ElasticNet |
| Large-scale data (>10,000 samples) | `SGDRegressor` with the right `penalty` |
| Small/medium data | `Ridge`, `Lasso`, or `RidgeCV`/`LassoCV` (sklearn's own recommendation) |
| Need to pick `alpha` automatically | `RidgeCV` / `LassoCV` / `ElasticNetCV`, or `GridSearchCV` |

Want me to build a small worked example (with a real dataset) showing how `alpha` changes the fitted coefficients and R² score, so you can see regularization's effect visually?
