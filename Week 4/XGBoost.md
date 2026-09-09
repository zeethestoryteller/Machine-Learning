# XGBoost Complete Guide: Estimators & Parameters

## 1. What XGBoost Actually Is

XGBoost (eXtreme Gradient Boosting) builds an ensemble of decision trees **sequentially**. Each new tree tries to correct the errors (residuals) of the trees built before it. This is different from Random Forest, where trees are built independently and averaged.

The core idea:
1. Start with a simple prediction (e.g., the mean of the target).
2. Compute the error (residual/gradient) of that prediction.
3. Build a new tree to predict that error.
4. Add this new tree's prediction (scaled by a learning rate) to the running total.
5. Repeat for `n_estimators` rounds.

Each tree is one "estimator." So `n_estimators` = number of boosting rounds = number of trees built.

---

## 2. Installation

```bash
pip install xgboost --break-system-packages
```

---

## 3. The Two APIs

XGBoost has two interfaces. Learn both — you'll see both in the wild.

### A. Scikit-learn API (easiest, recommended for most people)
```python
from xgboost import XGBClassifier, XGBRegressor
```

### B. Native XGBoost API (more control, used for advanced workflows)
```python
import xgboost as xgb
dtrain = xgb.DMatrix(X_train, label=y_train)
```

We'll focus mainly on the sklearn API since it's what you'll use 90% of the time, then show the native API too.

---

## 4. Full Working Example (Classification)

```python
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. Create sample data
X, y = make_classification(n_samples=2000, n_features=20, n_informative=15,
                            n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Define the model with explicit parameters
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0,
    reg_alpha=0,
    reg_lambda=1,
    min_child_weight=1,
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=20
)

# 3. Train with early stopping (needs an eval set)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# 4. Predict
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Best iteration:", model.best_iteration)
```

For regression, swap `XGBClassifier` → `XGBRegressor`, and `objective='binary:logistic'` → `objective='reg:squarederror'`.

---

## 5. Every Major Parameter — Explained

### 5.1 Core boosting control

| Parameter | Default | What it does | Why you tune it |
|---|---|---|---|
| `n_estimators` | 100 | Number of boosting rounds (trees) to build. | More trees = model can capture more complex patterns, but too many = overfitting and slower training. Usually paired with early stopping so XGBoost stops adding trees once validation performance stops improving. |
| `learning_rate` (`eta`) | 0.3 | Shrinks the contribution of each tree. Lower = each tree has less influence. | Lower learning rate + more trees = smoother, more generalizable model, at the cost of training time. Classic tradeoff: `learning_rate=0.01` with `n_estimators=1000` often beats `learning_rate=0.3` with `n_estimators=100`. |
| `max_depth` | 6 | Maximum depth of each tree. | Controls how complex a single tree can get. Deeper trees fit more intricate patterns but overfit more easily and are slower. Typical range: 3–10. |
| `min_child_weight` | 1 | Minimum sum of instance weight (roughly, minimum number of samples) needed in a leaf node to allow a further split. | Higher values make the algorithm more conservative — it won't split a node unless there's enough data to justify it. Prevents overfitting on small, noisy subsets. |

### 5.2 Regularization (prevents overfitting)

| Parameter | Default | What it does | Why you tune it |
|---|---|---|---|
| `gamma` (`min_split_loss`) | 0 | Minimum loss reduction required to make a further split on a leaf node. | Higher gamma = the tree only splits when it meaningfully reduces error. Acts as a pruning mechanism — kills splits that barely help. |
| `reg_alpha` | 0 | L1 regularization on leaf weights. | Encourages sparsity — pushes some leaf weights to exactly zero. Useful with high-dimensional/sparse data (many irrelevant features). |
| `reg_lambda` | 1 | L2 regularization on leaf weights. | Shrinks leaf weights smoothly toward zero (doesn't zero them out like L1). This is XGBoost's main defense against overfitting — increase it if the model overfits. |
| `max_delta_step` | 0 | Maximum absolute step each leaf's weight update can take in one round. | Mostly used for extremely imbalanced classification (logistic regression on rare-event data), where it stabilizes updates. Usually leave at 0. |

### 5.3 Sampling (adds randomness, like Random Forest does)

| Parameter | Default | What it does | Why you tune it |
|---|---|---|---|
| `subsample` | 1.0 | Fraction of training rows randomly sampled before growing each tree. | Values like 0.7–0.9 reduce overfitting and speed up training by not showing every tree all the data (similar idea to bagging). |
| `colsample_bytree` | 1.0 | Fraction of features randomly sampled for each tree. | Forces trees to be diverse — no single tree can rely on one dominant feature every time. Reduces overfitting, especially with many correlated features. |
| `colsample_bylevel` | 1.0 | Fraction of features sampled at each depth level within a tree. | Finer-grained than `colsample_bytree`. Rarely changed unless you have very high-dimensional data. |
| `colsample_bynode` | 1.0 | Fraction of features sampled at each individual split. | Even finer-grained control, similar spirit to Random Forest's per-split feature sampling. |

### 5.4 Objective, evaluation, and task-type

| Parameter | Default | What it does | Why you tune it |
|---|---|---|---|
| `objective` | `reg:squarederror` | The loss function to optimize. Examples: `binary:logistic` (binary classification, outputs probability), `multi:softmax` (multiclass, outputs class label), `multi:softprob` (multiclass, outputs probabilities), `reg:squarederror` (regression), `reg:logistic`, `rank:pairwise` (ranking). | Must match your problem type — this literally defines what the model is trying to minimize. |
| `eval_metric` | depends on objective | Metric used to evaluate on validation data during training (e.g., `logloss`, `auc`, `rmse`, `mae`, `merror`). | Lets you monitor training progress and drive early stopping using the metric you actually care about (e.g., AUC for imbalanced classification instead of accuracy). |
| `num_class` | — | Number of classes (required for `multi:softmax`/`multi:softprob`). | XGBoost needs to know how many output classes to build separate score paths for. |
| `base_score` | 0.5 | The initial prediction before any trees are added. | Rarely changed, but useful if you know the baseline rate (e.g., for a rare-event classifier, setting this near the true prevalence can speed convergence). |

### 5.5 Handling imbalanced data / missing values

| Parameter | Default | What it does | Why you tune it |
|---|---|---|---|
| `scale_pos_weight` | 1 | Balances positive/negative class weights in binary classification. Typically set to `count(negative)/count(positive)`. | Without this, a model trained on, say, 95%/5% imbalanced data will just predict the majority class. This rebalances the gradient contributions. |
| `missing` | `np.nan` | The value XGBoost treats as "missing." | XGBoost natively learns the best direction (left/right) to send missing values during a split — you usually don't need to impute missing data manually. |

### 5.6 Tree construction method (performance/scale)

| Parameter | Default | What it does | Why you tune it |
|---|---|---|---|
| `tree_method` | `auto` | Algorithm used to build trees: `exact` (precise, slow, small data), `approx`, `hist` (fast histogram-based, default on most modern setups), `gpu_hist`/`hist` with `device='cuda'` (GPU-accelerated). | For large datasets, `hist` is dramatically faster with negligible accuracy loss. Use GPU if you have one and a large dataset. |
| `grow_policy` | `depthwise` | `depthwise` grows trees level by level; `lossguide` grows wherever loss reduction is highest (like LightGBM's leaf-wise growth). | `lossguide` can be more accurate on complex data but risks deeper, more overfit-prone trees. |
| `max_leaves` | 0 (no limit) | Maximum number of leaf nodes (used with `grow_policy='lossguide'`). | Direct control over model complexity when growing leaf-wise instead of depth-wise. |
| `max_bin` | 256 | Number of bins used by the histogram algorithm to bucket continuous features. | Higher = more precise splits but slower and more memory. Lower = faster but coarser splits. |

### 5.7 Reproducibility & system

| Parameter | Default | What it does | Why you tune it |
|---|---|---|---|
| `random_state` (`seed`) | 0 | Random seed for reproducibility (sampling, etc.). | Without fixing this, results vary slightly between runs — important for debugging and fair comparisons. |
| `n_jobs` (`nthread`) | -1 (all cores) | Number of CPU threads used for training. | Set to -1 to use all available cores and speed up training; lower it if you need to share the machine. |
| `verbosity` | 1 | Logging level (0 = silent, 1 = warning, 2 = info, 3 = debug). | Useful for debugging training issues (e.g., seeing why a parameter combination fails). |
| `early_stopping_rounds` | None | Stops training if the validation metric hasn't improved for this many rounds. | Prevents overfitting and wastes no time training trees beyond the point of diminishing returns. Requires an `eval_set`. |

---

## 6. Full Parameter List in Code (with inline explanations)

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    # --- Core boosting ---
    n_estimators=500,        # number of trees to build
    learning_rate=0.05,      # shrinks each tree's contribution; lower = more robust, needs more trees
    max_depth=5,             # max depth per tree; controls model complexity

    # --- Regularization (overfitting control) ---
    min_child_weight=3,      # min samples required in a leaf to allow a split
    gamma=0.1,               # min loss reduction required to split further (pruning)
    reg_alpha=0.01,          # L1 regularization on leaf weights (sparsity)
    reg_lambda=1.0,          # L2 regularization on leaf weights (smoothness)

    # --- Sampling (randomness/generalization) ---
    subsample=0.8,           # fraction of rows sampled per tree
    colsample_bytree=0.8,    # fraction of features sampled per tree
    colsample_bylevel=1.0,   # fraction of features sampled per depth level
    colsample_bynode=1.0,    # fraction of features sampled per split

    # --- Task definition ---
    objective='binary:logistic',  # loss function to optimize
    eval_metric='auc',            # metric tracked during training/early stopping

    # --- Imbalance / missing data ---
    scale_pos_weight=1,      # rebalance classes; set to neg/pos ratio if imbalanced
    missing=None,            # value treated as "missing" (default NaN)

    # --- Tree construction / performance ---
    tree_method='hist',      # fast histogram-based tree building
    grow_policy='depthwise', # depthwise vs lossguide tree growth
    max_bin=256,             # number of histogram bins for splitting

    # --- Reproducibility / system ---
    random_state=42,         # reproducibility seed
    n_jobs=-1,                # use all CPU cores
    verbosity=1,               # logging level

    # --- Early stopping ---
    early_stopping_rounds=30  # stop if no improvement for 30 rounds (needs eval_set)
)
```

---

## 7. Native API Version (for comparison)

```python
import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'max_depth': 5,
    'eta': 0.05,               # same as learning_rate
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'lambda': 1.0,             # reg_lambda
    'alpha': 0.01,             # reg_alpha
    'min_child_weight': 3,
    'gamma': 0.1,
    'tree_method': 'hist'
}

bst = xgb.train(
    params,
    dtrain,
    num_boost_round=500,       # same as n_estimators
    evals=[(dtrain, 'train'), (dtest, 'eval')],
    early_stopping_rounds=30,
    verbose_eval=50
)

preds = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))
```

---

## 8. How to Actually Tune These (Practical Workflow)

Tuning 15+ parameters at once is unrealistic. Tune in this order:

1. **Fix a small learning rate isn't ideal for search** — instead, start with `learning_rate=0.1`, and a moderate `n_estimators=200` with early stopping on.
2. **Tune tree structure first**: `max_depth`, `min_child_weight` — these have the biggest impact on model capacity.
3. **Tune regularization**: `gamma`, `reg_alpha`, `reg_lambda` — reduce overfitting once you know the tree structure is reasonable.
4. **Tune sampling**: `subsample`, `colsample_bytree` — adds robustness.
5. **Lower the learning rate and increase n_estimators**, using early stopping to find the actual optimal number of trees at that lower rate. This final step usually gives the biggest accuracy boost.

```python
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'max_depth': [3, 4, 5, 6, 7, 8],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0, 0.1, 0.2, 0.3],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.01, 0.1, 1],
    'reg_lambda': [0.1, 1, 5, 10],
}

search = RandomizedSearchCV(
    XGBClassifier(n_estimators=200, learning_rate=0.1, random_state=42),
    param_distributions=param_dist,
    n_iter=50,
    scoring='roc_auc',
    cv=5,
    random_state=42,
    n_jobs=-1
)
search.fit(X_train, y_train)
print(search.best_params_)
```

---

## 9. Feature Importance (bonus — you'll want this)

```python
import matplotlib.pyplot as plt
from xgboost import plot_importance

plot_importance(model, max_num_features=15, importance_type='gain')
plt.show()
```
`importance_type` options: `'weight'` (number of times a feature is used to split), `'gain'` (average improvement in accuracy from splits on this feature — usually the most meaningful), `'cover'` (average number of samples affected by splits on this feature).

---

## 10. Quick-Reference Cheat Sheet

- **Model underfitting (low train + test accuracy)** → increase `max_depth`, `n_estimators`; decrease regularization (`reg_lambda`, `reg_alpha`, `gamma`).
- **Model overfitting (high train, low test accuracy)** → decrease `max_depth`; increase `min_child_weight`, `gamma`, `reg_lambda`, `reg_alpha`; decrease `subsample`/`colsample_bytree`; lower `learning_rate` and rely on early stopping.
- **Training too slow** → use `tree_method='hist'`, reduce `max_bin`, reduce `n_estimators`/increase `learning_rate`, use `n_jobs=-1`.
- **Imbalanced classes** → set `scale_pos_weight`.
