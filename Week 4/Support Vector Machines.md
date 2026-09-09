# Support Vector Machines (SVM) — Complete Guide
*Based on Dr. Ashish Tendulkar's IIT Madras "Machine Learning Practice" slides, with full explanations*

---

## 1. What is SVM?

Support Vector Machines are **supervised learning methods** used for:
- Classification
- Regression
- Outlier detection

**Core idea:** SVM tries to find a hyperplane (or set of hyperplanes) in a high- or even infinite-dimensional space that best separates classes. "Best" means the hyperplane that is placed as far as possible from the nearest points of each class — this distance is called the **margin**. The points closest to the boundary, which "support" this margin, are called **support vectors** — hence the name.

### The three sklearn implementations

| Class | Based on | Key trait |
|---|---|---|
| `SVC` | libsvm | Full-featured, supports all kernels |
| `NuSVC` | libsvm | Same as SVC, but controls margin errors via `nu` instead of `C` |
| `LinearSVC` | liblinear | Faster, linear kernel only, scales better |

**Why does this matter?** `SVC`/`NuSVC` use `libsvm`, which computes a full kernel matrix — this is powerful (any kernel, non-linear boundaries) but slow for large datasets (scales roughly O(n²) to O(n³)). `LinearSVC` uses `liblinear`, an optimizer built specifically for the linear case, so it scales near-linearly with data size. **Rule of thumb: if you know your data is linearly separable or you have a huge dataset, use `LinearSVC`. If you need non-linear boundaries, use `SVC`.**

---

## 2. Preparing Your Training Data

```python
X = [[0, 0], [1, 1]]   # Feature matrix: shape (n_samples, n_features)
y = [0, 1]             # Labels: shape (n_samples,)
```

- **`X`**: one row per sample, one column per feature.
- **`y`**: class labels — can be integers or strings.

**Why this matters:** SVM (like most sklearn estimators) is strictly array-in, array-out. Getting the shapes right (2D `X`, 1D `y`) avoids the most common beginner errors.

---

## 3. SVC — C-Support Vector Classification

### Step 1: Instantiate
```python
from sklearn.svm import SVC
SVC_classifier = SVC()
```

### Step 2: Fit
```python
SVC_classifier.fit(X_train, y_train)
```

That's the whole workflow — instantiate, then `.fit(X, y)`. Predictions later come from `.predict(X_test)`.

### Key Parameters of SVC (explained)

#### `C` — Regularization parameter
```python
SVC_classifier = SVC(C=1.0)   # default
```
- **What it does:** Controls the trade-off between a smooth decision boundary and classifying training points correctly.
- **Why it matters:** SVM's objective has two competing goals — maximize the margin AND minimize classification error on training data. `C` weighs how much you penalize misclassified/margin-violating points.
  - **Large C** → less regularization → model tries hard to classify every training point correctly → narrow margin → risk of **overfitting**.
  - **Small C** → more regularization → model allows more margin violations → wider margin → smoother boundary → risk of **underfitting**.
- Must be **strictly positive** (a float).
- The penalty used is a **squared L2 penalty** (it penalizes the squared magnitude of the margin violations, not just their count).
- **Practical tip:** Always tune `C` via cross-validation (e.g., `GridSearchCV` over `[0.01, 0.1, 1, 10, 100]`) — it's the single most important knob in SVC.

#### `kernel` — how similarity between points is computed
```python
SVC_classifier = SVC(kernel='rbf')   # default
```
Options: `'linear'`, `'poly'`, `'rbf'`, `'sigmoid'`, `'precomputed'`, or a callable.

- **Why kernels exist:** Many datasets aren't linearly separable in their original feature space. Kernels implicitly map data into a higher-dimensional space where a linear separator *does* exist — without ever computing that mapping explicitly (the "kernel trick"), which saves enormous computation.
- **`'linear'`** — no transformation; use when you believe classes are linearly separable, or you have very high-dimensional data (e.g., text/TF-IDF) where linear already works well.
- **`'poly'`** — polynomial kernel; captures curved/interaction boundaries. Requires setting **`degree`** (an integer — degree of the polynomial. Higher degree = more flexible boundary, but higher overfitting risk and cost).
- **`'rbf'`** (Radial Basis Function / Gaussian) — the default and most commonly effective kernel for non-linear data. Creates smooth, localized decision boundaries.
- **`'sigmoid'`** — resembles a neural network activation; occasionally useful, less common in practice.
- **`'precomputed'`** — you supply the kernel (Gram) matrix yourself instead of raw features. Useful in advanced/custom-similarity setups.
- **callable** — pass your own function to compute the kernel matrix from data matrices, for full custom control.

#### `gamma` — kernel coefficient (for `'rbf'`, `'poly'`, `'sigmoid'`)
```python
SVC_classifier = SVC(gamma='scale')   # default
```
- **What it does:** Controls how far the influence of a single training example reaches.
  - **High gamma** → each point's influence is very local/narrow → boundary hugs the training points tightly → **overfitting risk**.
  - **Low gamma** → influence reaches far → boundary is smoother/more general → **underfitting risk** if too low.
- **`'scale'`** (default): `gamma = 1 / (n_features * X.var())` — automatically adapts to your feature scale/variance. Generally a safe default.
- **`'auto'`**: `gamma = 1 / n_features` — ignores variance of the data.
- **float value**: you can also supply your own numeric gamma directly and tune it via cross-validation.
- **Why it matters:** `gamma` and `C` interact — together they control the bias-variance trade-off of an RBF-kernel SVM. In practice, both are grid-searched together (`GridSearchCV` over a `C` × `gamma` grid).

#### `coef0` — independent term (for `'poly'` and `'sigmoid'` kernels)
```python
SVC_classifier = SVC(kernel='poly', degree=3, coef0=1)
```
- **What it does:** Shifts the kernel function — controls how much influence higher-degree vs. lower-degree terms have in the polynomial/sigmoid kernel.
- **Why it matters:** Without it, the kernel's behavior near zero can be poorly conditioned; adjusting `coef0` gives you another lever to shape the decision boundary. Any integer (or float) value can be used; it's often left at default (0) unless tuning shows benefit.

### Viewing Support Vectors (after fitting)
```python
from sklearn.svm import SVC
SVC_classifier = SVC()
clf = SVC_classifier.fit(X_train, y_train)

clf.support_          # indices of the support vectors in X_train
clf.support_vectors_  # the actual support vector data points
clf.n_support_        # number of support vectors per class
```
**Why this is useful:** Inspecting support vectors tells you which training points are actually "doing the work" of defining the boundary — useful for interpretability, debugging class imbalance, and understanding model complexity (fewer support vectors generally = simpler, more generalizable model).

---

## 4. NuSVC — ν-Support Vector Classification

### Code
```python
from sklearn.svm import NuSVC
NuSVC_classifier = NuSVC()
NuSVC_classifier.fit(X_train, y_train)
```

### Key Parameter: `nu`
```python
NuSVC_classifier = NuSVC(nu=0.5)   # default
```
- **What it replaces:** Instead of `C` (an unbounded regularization strength), NuSVC uses `nu`, which has a direct, interpretable meaning:
  - `nu` is an **upper bound on the fraction of margin errors** (points on the wrong side of the margin or misclassified).
  - `nu` is also a **lower bound on the fraction of support vectors**.
- **Range:** must be in `(0, 1]`.
- **Why use it over `C`:** `nu` is more interpretable — e.g., `nu=0.05` means "I expect at most ~5% of my training points to be margin violations." This is handy when you have a rough sense of acceptable error rate but don't have intuition for what a "good" `C` value looks like.
- All other parameters (`kernel`, `gamma`, `coef0`, etc.) work identically to `SVC`.

---

## 5. LinearSVC — Linear Support Vector Classification

### Code
```python
from sklearn.svm import LinearSVC
LinearSVC_classifier = LinearSVC()
LinearSVC_classifier.fit(X_train, y_train)
```

### Why use LinearSVC instead of SVC(kernel='linear')?
- More flexible choice of **penalties and loss functions** (built on `liblinear`).
- **Scales much better to large numbers of samples** — this is the big one. `SVC` with a linear kernel still runs through the general libsvm machinery; `LinearSVC` uses an optimizer designed specifically for the linear case.
- Supports both **dense and sparse** input (sparse matters a lot for text/TF-IDF data with many zero entries).

### Key Parameters

#### `penalty`
```python
LinearSVC_classifier = LinearSVC(penalty='l2')   # default
```
- **`'l2'`** — standard ridge-style penalty; keeps all feature weights small but non-zero.
- **`'l1'`** — leads to **sparse `coef_` vectors** (many weights become exactly zero). 
- **Why it matters:** Use `'l1'` when you want automatic **feature selection** built into training — irrelevant features get zeroed out, giving you a simpler, more interpretable model. Use `'l2'` when you want to keep all features but shrink their influence.

#### `loss`
```python
LinearSVC_classifier = LinearSVC(loss='squared_hinge')   # default
```
- **`'hinge'`** — the standard SVM loss, `max(0, 1 - y·f(x))`. Linear penalty for margin violations.
- **`'squared_hinge'`** — squares the hinge loss, `max(0, 1 - y·f(x))²`. Penalizes larger violations disproportionately more, and is smoother (differentiable), which often helps optimization converge better/faster.
- **Important restriction:** the combination `penalty='l1'` + `loss='hinge'` is **not supported** in sklearn — if you need L1 penalty, you must use `loss='squared_hinge'`.

#### `C` — same regularization role as in SVC
Smaller `C` = more regularization = wider margin, more tolerance for misclassification. Larger `C` = fits training data harder.

#### `dual`
```python
LinearSVC_classifier = LinearSVC(dual=True)  # (older default; check your sklearn version)
```
- **What it does:** Chooses whether to solve the **dual** or **primal** optimization formulation of the SVM objective.
- **Why it matters (performance):** 
  - **When `n_samples > n_features`** → prefer `dual=False` (solving the primal is faster/more efficient in this regime).
  - When `n_features > n_samples` (e.g., text data with huge vocabularies), the dual formulation is typically more efficient.
- This is a pure computational-efficiency choice — it does not change what the model learns, only how fast/stably it's found.

#### `fit_intercept`
```python
LinearSVC_classifier = LinearSVC(fit_intercept=True)   # default
```
- **What it does:** Whether to calculate a bias/intercept term for the decision function (i.e., allow the separating hyperplane to not pass through the origin).
- **Why it matters:** Almost always keep this `True` unless your data is already centered at zero and you have a specific reason to force the hyperplane through the origin.

---

## 6. Multi-class Classification with SVM

SVM is inherently binary, so multi-class support is built via strategies:

| Class | Strategy | Controlling parameter |
|---|---|---|
| `SVC`, `NuSVC` | **One-vs-One (OvO)** — trains a classifier for every pair of classes | `decision_function_shape`: `'ovo'` or `'ovr'` |
| `LinearSVC` | **One-vs-Rest (OvR)** — trains one classifier per class vs. all others | `multi_class`: `'ovr'` or `'crammer_singer'` |

- **Why OvO for SVC/NuSVC:** libsvm is naturally built around pairwise classification; for `k` classes this trains `k(k-1)/2` classifiers. `decision_function_shape` only changes how these pairwise results are **combined into an output shape** — it doesn't change how many models are trained internally.
- **Why OvR for LinearSVC:** it's simpler and scales better — only `k` classifiers needed for `k` classes.
- **`'crammer_singer'`**: an alternative multi-class formulation that optimizes all classes jointly rather than independently — theoretically elegant but rarely used in practice since it's often no better and less efficient.

---

## 7. Complete Worked Example

```python
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# 1. Load & split data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Scale features — SVM is distance-based, so scaling matters A LOT
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Instantiate and tune hyperparameters
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 0.01, 0.1, 1],
    'kernel': ['rbf', 'linear']
}
grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
best_model = grid.best_estimator_

# 4. Evaluate
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))

# 5. Inspect support vectors
print("Number of support vectors per class:", best_model.n_support_)
```

**Why scaling matters (important, not in the slides but critical in practice):** SVM decision boundaries depend on distances between points (directly for linear kernels, and via the kernel function for RBF/poly). Features on different scales (e.g., age in years vs. income in dollars) will distort these distances, letting large-scale features dominate. **Always scale your features (e.g., `StandardScaler`) before fitting an SVM.**

---

## 8. Advantages of SVM
- **Effective in high-dimensional spaces** — works well even when you have many features.
- **Effective when `n_features > n_samples`** — unlike many models, it doesn't automatically break down here.
- **Memory efficient** — the decision function only depends on a subset of training points (the support vectors), not the whole dataset.
- **Versatile** — different kernel functions let you adapt to linear or non-linear problems.

## 9. Disadvantages of SVM
- **No direct probability estimates** — sklearn computes them via an expensive internal 5-fold cross-validation (`probability=True` in SVC), which slows down training.
- **Overfitting risk when `n_features >> n_samples`** — careful kernel and regularization choice is essential in this regime.
- **Doesn't scale as well as some models to very large datasets** (except `LinearSVC`, which is designed for that case).

---

## 10. Quick Decision Guide

| Situation | Recommendation |
|---|---|
| Large dataset, suspect linear separability | `LinearSVC` |
| Need non-linear boundary, dataset is small/medium | `SVC(kernel='rbf')` |
| Want interpretable "expected error rate" control | `NuSVC` |
| Need feature selection built-in | `LinearSVC(penalty='l1', loss='squared_hinge')` |
| Need probability outputs | `SVC(probability=True)` (slower) |
| High-dimensional sparse data (text) | `LinearSVC` (handles sparse input well) |

---

### Summary of all parameters covered

| Parameter | Where | Purpose |
|---|---|---|
| `C` | SVC, LinearSVC | Regularization strength (error tolerance vs. margin width) |
| `nu` | NuSVC | Bounds fraction of margin errors / support vectors |
| `kernel` | SVC, NuSVC | Similarity function shaping the decision boundary |
| `degree` | SVC (poly) | Flexibility of polynomial boundary |
| `gamma` | SVC, NuSVC | Reach of a single point's influence (rbf/poly/sigmoid) |
| `coef0` | SVC (poly/sigmoid) | Shifts kernel function behavior |
| `penalty` | LinearSVC | L1 (sparse) vs L2 (dense) weight regularization |
| `loss` | LinearSVC | hinge vs squared_hinge loss function |
| `dual` | LinearSVC | Solve dual vs primal — efficiency choice based on n_samples vs n_features |
| `fit_intercept` | LinearSVC | Whether to learn a bias term |
| `decision_function_shape` | SVC, NuSVC | ovo vs ovr output shape for multi-class |
| `multi_class` | LinearSVC | ovr vs crammer_singer strategy for multi-class |
