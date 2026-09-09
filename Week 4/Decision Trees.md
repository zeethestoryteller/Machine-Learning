# Decision Trees — Complete Guide (Theory + Code)

## 1. What Is a Decision Tree?

A decision tree is a **non-parametric supervised learning** method that can learn both **classification** and **regression** models. It predicts a label by following **rules** inferred from the **features** in the training data — essentially a flowchart of if/else questions that ends in a prediction.

- "Non-parametric" means the model doesn't assume a fixed functional form (like a line in linear regression) — its complexity grows with the data.
- It works by recursively splitting the dataset into purer and purer subsets based on feature values.

## 2. Tree-Building Algorithms

There have been several historical algorithms for building trees:

| Algorithm | Key Idea |
|---|---|
| **ID3** (Iterative Dichotomiser 3) | Creates a **multiway tree** (a node can split into more than 2 branches) |
| **C4.5** | Successor to ID3; converts the trained tree into **sets of if-then rules** |
| **C5.0** | Quinlan's latest version, released under a **proprietary license**; uses less memory and builds smaller rule sets |
| **CART** (Classification And Regression Trees) | Supports **numerical target variables** (regression) and does **not** compute rule sets — always produces **binary splits** |

**scikit-learn uses an optimized version of CART.** One important limitation: sklearn's implementation does **not support categorical variables natively** — you must encode them first (one-hot encoding, ordinal encoding, etc.).

## 3. The Two Estimators in sklearn

```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
```

| Task | Class |
|---|---|
| Classification | `sklearn.tree.DecisionTreeClassifier` |
| Regression | `sklearn.tree.DecisionTreeRegressor` |

Both share the **same parameters** except for `criterion`, which differs because classification and regression measure "split quality" differently.

## 4. Parameters — What They Do and Why You Use Them

### `splitter` — strategy used at each node
- **`"best"`** (default): evaluates all possible splits and picks the best one. Use this when you want the most accurate, deterministic tree.
- **`"random"`**: picks the best split among a **random subset** of features/thresholds. Use this to add randomness (useful for reducing overfitting or when building ensembles like Extra-Trees) and to speed up training on large datasets.

**Why it matters:** `"best"` gives you the most predictive single tree; `"random"` trades a bit of accuracy for speed and variance reduction.

### `max_depth` — maximum depth of the tree
- An `int`, or `None`.
- If `None`, the tree keeps expanding **until all leaves are pure** (contain only one class / a single value) **or** until a node has fewer samples than `min_samples_split`.

**Why it matters:** This is your #1 lever against overfitting. An unconstrained tree (`max_depth=None`) will happily memorize the training set, creating a very deep tree with poor generalization. Limiting depth forces the tree to keep only the most important splits.

### `min_samples_split` — minimum samples to split an internal node
- `int` or `float` (fraction of total samples). **Default: 2.**
- A node needs at least this many samples before the algorithm is even allowed to try splitting it further.

**Why it matters:**
- **Too small** (e.g., the default 2) → the tree will keep splitting down to very small, noisy groups → **overfitting**.
- **Too large** → the tree stops splitting early and can't capture real patterns → **underfitting** (tree fails to learn the data).

### `min_samples_leaf` — minimum samples required to be at a leaf
- `int` or `float`. **Default: 1.**
- Ensures every leaf (final prediction node) is supported by at least this many training samples.

**Why it matters:** Same overfitting/underfitting trade-off as above, but applied to leaves specifically. Using `min_samples_split` or `min_samples_leaf` ensures that **multiple samples influence every decision** in the tree, by controlling which splits are even considered:
- A **very small** value → the tree overfits (single outliers can create their own leaf).
- A **large** value → the tree can't learn fine-grained patterns in the data.

### `criterion` — function to measure split quality
This is the one parameter that differs between classifier and regressor:

**Classification:**
- **`gini`** (default): Gini impurity — measures how often a randomly chosen sample would be misclassified if labeled according to the class distribution in that node. Faster to compute.
- **`entropy`**: Information gain based on Shannon entropy — measures the "disorder" in a node. Slightly more computationally expensive, sometimes produces more balanced trees.

**Regression:**
- **`squared_error`** (default): minimizes variance (MSE) within each node — best for typical continuous targets.
- **`friedman_mse`**: a variant of MSE with an improvement score from Friedman, often used in gradient boosting.
- **`absolute_error`**: minimizes MAE — more robust to outliers than squared error.
- **`poisson`**: use when the target is **count data** (non-negative, Poisson-distributed).

**Why it matters:** The criterion decides *how* the algorithm evaluates "which split is best" at every node. Choosing the right one aligns the tree's objective with your data type (e.g., use `poisson` for count targets, `absolute_error` if you have outliers).

## 5. Full Code Example

```python
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# 1. Load / prepare data
# X: features (must be numeric — encode categorical columns first!)
# y: target labels
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Instantiate and train the model
clf = DecisionTreeClassifier(
    criterion="gini",        # or "entropy"
    splitter="best",         # or "random"
    max_depth=5,              # start shallow, increase as needed
    min_samples_split=10,     # require 10+ samples to split a node
    min_samples_leaf=5,       # require 5+ samples in every leaf
    random_state=42
)
clf.fit(X_train, y_train)

# 3. Evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 4. Visualize the tree
plt.figure(figsize=(16, 8))
plot_tree(
    clf,
    max_depth=3,                     # start with a shallow view, then increase
    feature_names=X.columns,
    class_names=[str(c) for c in clf.classes_],
    filled=True,
    label="all"
)
plt.show()
```

### Regression version

```python
from sklearn.tree import DecisionTreeRegressor

reg = DecisionTreeRegressor(
    criterion="squared_error",   # or "friedman_mse", "absolute_error", "poisson"
    splitter="best",
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
reg.fit(X_train, y_train)
```

## 6. Avoiding Overfitting: Pre-pruning vs. Post-pruning

### Pre-pruning
Search over hyperparameters (`max_depth`, `min_samples_split`, `min_samples_leaf`, etc.) using `GridSearchCV` to find the best combination **before** the tree is fully grown.

```python
param_grid = {
    "max_depth": [3, 5, 7, 10, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 5, 10],
    "criterion": ["gini", "entropy"]
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy"
)
grid_search.fit(X_train, y_train)

print("Best params:", grid_search.best_params_)
best_tree = grid_search.best_estimator_
```

### Post-pruning
Grow the tree **fully** (no constraints), then prune it back using **cost-complexity pruning** (controlled by the `ccp_alpha` parameter).

```python
# Grow an unconstrained tree first
full_tree = DecisionTreeClassifier(random_state=42)
full_tree.fit(X_train, y_train)

# Get the pruning path (sequence of alpha values)
path = full_tree.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# Train a tree for each alpha and pick the best on validation/test performance
trees = []
for alpha in ccp_alphas:
    t = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
    t.fit(X_train, y_train)
    trees.append(t)

test_scores = [accuracy_score(y_test, t.predict(X_test)) for t in trees]
best_alpha_idx = test_scores.index(max(test_scores))
best_pruned_tree = trees[best_alpha_idx]
```

## 7. `plot_tree` Parameters

```python
sklearn.tree.plot_tree(
    decision_tree,      # the trained tree to plot
    max_depth=None,     # how many levels to display; None = full tree
    feature_names=None, # names shown for each feature (readability)
    class_names=None,   # names for target classes, in ascending numeric order
    label="all"         # whether to show informative impurity labels
)
```

Tip: start with `max_depth=3` to get an initial feel for how the tree is splitting, then increase depth once you understand the top-level logic.

## 8. Practical Tips (from the slides)

1. **Watch the samples-to-features ratio.** Trees overfit easily when there are many features relative to the number of samples.
2. **Do dimensionality reduction first** (PCA or feature selection) — this improves the odds of finding truly discriminative features.
3. **Visualize progressively** — plot with `max_depth=3` first, then increase depth to inspect deeper splits.
4. **Balance your dataset** before training, so the tree doesn't become biased toward majority classes.

## 9. Quick Mental Model / Summary

| Goal | Parameter to tune |
|---|---|
| Control overall tree size / prevent runaway overfitting | `max_depth` |
| Prevent splits on tiny, noisy subsets | `min_samples_split` |
| Ensure statistically meaningful leaves | `min_samples_leaf` |
| Change how "purity" of a split is measured | `criterion` |
| Trade some accuracy for speed/randomness | `splitter="random"` |
| Systematically search for best hyperparameters | `GridSearchCV` (pre-pruning) |
| Prune an already-grown tree | `cost_complexity_pruning_path` + `ccp_alpha` (post-pruning) |

**Rule of thumb workflow:**
1. Fit an unconstrained tree to see baseline performance (usually overfits).
2. Visualize with `max_depth=3` to understand top splits.
3. Use `GridSearchCV` for pre-pruning **or** cost-complexity pruning for post-pruning.
4. Compare train vs. test accuracy — a big gap means overfitting, so tighten `max_depth` / raise `min_samples_leaf`.
