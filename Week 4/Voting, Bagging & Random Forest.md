# Voting, Bagging & Random Forest — Complete Guide

This guide walks through every estimator from the slides — **VotingClassifier/Regressor**, **BaggingClassifier/Regressor**, and **RandomForestClassifier/Regressor** — with runnable code and a plain-English explanation of *every* parameter: what it does and why you'd tune it.

We'll use one consistent dataset throughout so you can copy-paste and run top to bottom.

```python
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split

# Classification dataset
X_clf, y_clf = load_breast_cancer(return_X_y=True)
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)

# Regression dataset
X_reg, y_reg = load_diabetes(return_X_y=True)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)
```

---

## 1. Voting Estimators

**Idea:** Train several *different* models (e.g. a logistic regression, a tree, a KNN) independently on the *same* full dataset, then combine their predictions. This reduces variance and often bias because different model families make different kinds of mistakes — when they disagree, the ensemble averages the error out.

### 1.1 VotingClassifier

```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

clf1 = LogisticRegression(max_iter=5000)
clf2 = DecisionTreeClassifier(max_depth=5, random_state=42)
clf3 = KNeighborsClassifier(n_neighbors=5)

voting_clf = VotingClassifier(
    estimators=[('lr', clf1), ('dt', clf2), ('knn', clf3)],
    voting='soft',
    weights=[2, 1, 1],
    n_jobs=-1
)

voting_clf.fit(Xc_train, yc_train)
print("Accuracy:", voting_clf.score(Xc_test, yc_test))
```

**Parameters:**

| Parameter | What it does | Why you'd use it |
|---|---|---|
| `estimators` | List of `(name, estimator)` tuples — the individual models to combine. | This is the core of voting: you deliberately pick *diverse* models so their errors are uncorrelated. Similar models voting together doesn't help much. |
| `voting` (`'hard'` or `'soft'`) | `'hard'`: majority vote on predicted class labels. `'soft'`: averages predicted class *probabilities* and picks the highest. | `'soft'` usually performs better because it uses confidence, not just the winning label — but it requires every base estimator to implement `predict_proba`. Use `'hard'` if some estimators don't support probabilities. |
| `weights` | A list of weights, one per estimator, used when averaging (soft) or counting votes (hard). | Give more influence to your strongest models instead of treating a weak model equally with a strong one. |
| `n_jobs` | Number of CPU cores to fit estimators in parallel (`-1` = all cores). | Speeds up training since each base estimator is independent — pure engineering convenience. |
| `flatten_transform` | Controls the shape of the array returned by `transform()` when `voting='soft'`. | Only matters if you're chaining this into a `Pipeline` and need a specific 2D array shape downstream. Rarely touched. |
| `verbose` | Prints progress while fitting. | Debugging/monitoring on large datasets. |

Note: the slide lists `base_estimator` as a "common parameter" of voting estimators — in current scikit-learn, voting estimators actually take `estimators` (a list), not a single `base_estimator`. `base_estimator` is the Bagging-style parameter (below). I'm flagging this so you use the correct API.

### 1.2 VotingRegressor

```python
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

reg1 = Ridge(alpha=1.0)
reg2 = DecisionTreeRegressor(max_depth=5, random_state=42)
reg3 = SVR(kernel='rbf')

voting_reg = VotingRegressor(
    estimators=[('ridge', reg1), ('dt', reg2), ('svr', reg3)],
    weights=[2, 1, 1],
    n_jobs=-1
)

voting_reg.fit(Xr_train, yr_train)
print("R^2:", voting_reg.score(Xr_test, yr_test))
```

Same parameters as `VotingClassifier` minus `voting` (there's no hard/soft choice — regression predictions are simply averaged, optionally weighted).

**Common functions on both:** `fit`, `predict`, `fit_transform`, `score` — standard scikit-learn estimator API (`fit_transform`/`transform` returns the underlying predictions of each base estimator, useful for stacking or inspection).

---

## 2. Bagging Estimators

**Idea (Bootstrap AGGregatING):** Instead of using different *model types*, use *many copies of the same model type*, each trained on a different **random bootstrap sample** (sampling with replacement) of the training data. Averaging/voting over these reduces **variance** — this is why bagging helps most with high-variance, low-bias models like unpruned decision trees.

### 2.1 BaggingClassifier

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(),   # base_estimator in older sklearn versions
    n_estimators=100,
    max_samples=0.8,
    max_features=0.8,
    bootstrap=True,
    bootstrap_features=False,
    oob_score=True,
    n_jobs=-1,
    random_state=42
)

bag_clf.fit(Xc_train, yc_train)
print("Accuracy:", bag_clf.score(Xc_test, yc_test))
print("OOB score:", bag_clf.oob_score_)
```

### 2.2 BaggingRegressor

```python
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

bag_reg = BaggingRegressor(
    estimator=DecisionTreeRegressor(),
    n_estimators=100,
    max_samples=0.8,
    max_features=1.0,
    bootstrap=True,
    oob_score=True,
    random_state=42
)

bag_reg.fit(Xr_train, yr_train)
print("R^2:", bag_reg.score(Xr_test, yr_test))
print("OOB score:", bag_reg.oob_score_)
```

**Parameters (shared by both, matching the slides):**

| Parameter | Default | What it does | Why you'd use it |
|---|---|---|---|
| `base_estimator` (now `estimator`) | `None` (→ `DecisionTreeClassifier`/`Regressor`) | The model type to be replicated and trained on each bootstrap subset. | Bagging works with *any* base learner, but it helps most with unstable, high-variance learners (deep trees). Bagging a stable low-variance model like linear regression barely changes anything. |
| `n_estimators` | `10` | Number of base estimators (bags) to train. | More estimators → more variance reduction and smoother predictions, at the cost of training/prediction time. Returns diminish after some point (check via validation curve). |
| `max_samples` | `1.0` | Number (int) or fraction (float) of samples drawn **with replacement** (bootstrap, since `bootstrap=True` by default) to train each base estimator. | Controls how much each bag "sees." Smaller values increase diversity among bags (more variance reduction) but each individual tree gets less data (may increase bias). |
| `max_features` | `1.0` | Number/fraction of features drawn **without replacement by default** to train each base estimator. | Restricting features per bag adds another source of diversity (this is what turns Bagging into the "Random Subspace" method, and is the seed idea behind Random Forest). |
| `bootstrap` | `True` | Whether samples are drawn *with* replacement. | `True` = classic bagging (some rows repeated, some left out — those left-out rows become the OOB set). Set `False` to instead do random subsampling without replacement ("pasting"). |
| `bootstrap_features` | `False` | Whether features are drawn *with* replacement. | Usually left `False`; setting `True` lets the same feature be picked more than once per bag — rarely useful, occasionally used for very high-dimensional data. |
| `oob_score` | `False` | If `True`, uses the ~37% of samples **not** selected in a given bootstrap draw (the "out-of-bag" samples) to estimate generalization performance, without needing a separate validation set. | Free, near-unbiased validation score using only training data — very handy when data is limited. Access it via `.oob_score_` after fitting. |
| `n_jobs` | `None` | Parallelism across base estimators. | Speed only — each bag is trained independently. |
| `random_state` | `None` | Seeds the bootstrap sampling. | Reproducibility. |
| `warm_start` | `False` | If `True`, reuses the previously fitted ensemble and adds more estimators on the next `fit()` call rather than starting over. | Useful for incrementally growing the ensemble (e.g. tuning `n_estimators` without refitting from scratch). |

**Functions:** `fit`, `predict`, `score` — same standard API. `estimators_` (after fitting) holds the list of trained base learners, and `estimators_samples_` / `estimators_features_` tell you exactly which rows/columns each bag used.

---

## 3. Random Forest Estimators

**Idea:** Random Forest = Bagging **specifically with decision trees**, plus one extra trick: at *each split* inside each tree, only a random subset of features is even considered (not just a random subset per whole tree). This decorrelates the trees further than plain bagging does, which reduces variance even more.

So its parameters are literally the union of:
- **Bagging parameters** (how bootstrap samples are drawn)
- **Decision tree parameters** (how each individual tree is grown)

### 3.1 RandomForestClassifier

```python
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(
    # --- Bagging-style parameters ---
    n_estimators=200,
    bootstrap=True,
    oob_score=True,
    max_samples=0.8,
    n_jobs=-1,
    random_state=42,
    # --- Decision tree parameters ---
    criterion='gini',
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features='sqrt',
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    ccp_alpha=0.0,
    class_weight='balanced'
)

rf_clf.fit(Xc_train, yc_train)
print("Accuracy:", rf_clf.score(Xc_test, yc_test))
print("OOB score:", rf_clf.oob_score_)
print("Top feature importances:", rf_clf.feature_importances_[:5])
```

### 3.2 RandomForestRegressor

```python
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(
    n_estimators=200,
    bootstrap=True,
    oob_score=True,
    max_samples=0.8,
    n_jobs=-1,
    random_state=42,
    criterion='squared_error',
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=1.0,     # note: regressor default differs from classifier
    ccp_alpha=0.0
)

rf_reg.fit(Xr_train, yr_train)
print("R^2:", rf_reg.score(Xr_test, yr_test))
print("OOB score:", rf_reg.oob_score_)
```

### Bagging-style parameters (as in the slides)

| Parameter | Default | What it does | Why you'd use it |
|---|---|---|---|
| `n_estimators` | `100` for classifier and regressor (slide says 10/100 — modern sklearn defaults to 100 for both) | Number of trees in the forest. | More trees = lower variance, smoother predictions, more stable `feature_importances_`. Cost: more memory/compute. Track validation error vs. `n_estimators` — it plateaus. |
| `bootstrap` | `True` | `True`: each tree trained on a bootstrapped sample. `False`: every tree sees the *whole* dataset (only feature-level randomness remains). | Keep `True` for the classic Random Forest behavior and to enable OOB estimation. |
| `oob_score` | `False` | Whether to compute the out-of-bag accuracy/R² estimate. Only valid when `bootstrap=True`. | Gives you a "free" validation metric without a held-out set — great for small datasets. |
| `max_samples` | `None` (all samples) | Same as Bagging: how many rows per bootstrap draw. Only used when `bootstrap=True`. | Lower it to increase tree diversity / reduce overfitting, or to speed up training on huge datasets. |

### Decision tree parameters (control each individual tree)

| Parameter | Default | What it does | Why you'd use it |
|---|---|---|---|
| `criterion` | `'gini'` (classifier) / `'squared_error'` (regressor) | The impurity/loss function used to choose the best split at each node. Classification alternatives: `'entropy'`, `'log_loss'`. Regression alternatives: `'absolute_error'`, `'friedman_mse'`, `'poisson'`. | `gini`/`squared_error` are fast and usually fine defaults. `entropy` can sometimes give slightly different (more "balanced") splits. `absolute_error` is more robust to outliers in regression but much slower to compute. |
| `max_depth` | `None` | Maximum depth a tree can grow to. `None` = grow until leaves are pure or hit `min_samples_split`. | Limiting depth is the single strongest lever against overfitting for an individual tree. Since RF already averages many trees, this matters less than for a lone tree, but still helps with very noisy data or speed. |
| `min_samples_split` | `2` | Minimum number of samples an internal node must have before it's allowed to split further. Int = exact count, float = fraction of `n` samples. | Higher values stop the tree from creating splits based on tiny, noisy subgroups — a regularizer. |
| `min_samples_leaf` | `1` | Minimum number of samples required to be in a *leaf* node. | Prevents leaves that represent just 1–2 training points (which just memorize noise). Raising this smooths the decision boundary. |
| `min_weight_fraction_leaf` | `0.0` | Like `min_samples_leaf`, but expressed as a fraction of the total sum of sample weights, useful when samples aren't equally weighted. | Use when you're passing `sample_weight` to `fit()` (e.g. imbalanced classes) and want the leaf-size constraint to respect those weights rather than raw counts. |
| `max_features` | `'sqrt'` (classifier) / `1.0`, i.e. all features (regressor) | Number of features randomly considered at **each split**. Accepts `'sqrt'`, `'log2'`, an int, a float, or `None`/`1.0` (all features). This is the extra randomness that separates Random Forest from plain Bagging. | The whole point of Random Forest: forcing each split to consider a random subset of features decorrelates the trees (they can't all rely on the one dominant feature), which reduces the ensemble's variance more than bagging alone. `sqrt` is the classic classification default; regression tasks often benefit from considering more features. |
| `max_leaf_nodes` | `None` | Caps the total number of leaf nodes in a tree; trees grow in "best-first" order, always splitting the node that most reduces impurity next. | An alternative way to control tree size/complexity, sometimes more intuitive/faster to tune than `max_depth`. |
| `min_impurity_decrease` | `0.0` | A split is only performed if it decreases impurity by at least this amount. | A direct regularizer — stops splits that provide negligible predictive benefit, effectively pruning "useless" splits before they're even made. |
| `ccp_alpha` | `0.0` | Complexity parameter for **minimal cost-complexity pruning**: after growing, sub-trees are pruned away if their added complexity outweighs their impurity reduction, controlled by this alpha. | A more principled way to prune than fiddling with `max_depth`/`min_samples_leaf` individually — larger `ccp_alpha` = more aggressive pruning = simpler trees. |
| `class_weight` (classifier only) | `None` | Reweights classes (`'balanced'` auto-weights inversely to class frequency, or pass a dict) when computing impurity and predictions. | Essential for imbalanced classification (e.g. rare disease detection) — without it, the majority class dominates every split. |

### Attributes and methods after training

| Name | What it gives you |
|---|---|
| `estimators_` | The list of the actual fitted `DecisionTreeClassifier`/`Regressor` objects that make up the forest — you can inspect, plot, or evaluate any single tree. |
| `feature_importances_` | An array (summing to 1) showing how much each feature contributed to reducing impurity across all trees — your go-to for quick feature-importance ranking. |
| `oob_score_` | The out-of-bag accuracy/R² (only if `oob_score=True`). |
| `fit(X, y)` | Builds the forest given training data and the parameters above. |
| `predict(X)` | Class label (classifier) or numeric value (regressor) — majority vote / average across all trees. |
| `predict_proba(X)` / `predict_log_proba(X)` | (Classifier only) class probabilities / their logs, averaged across trees. |
| `decision_path(X)` | Returns which nodes of which trees each sample passed through — useful for interpretability/debugging. |

---

## 4. How the Three Relate (Mental Model)

```
VotingClassifier/Regressor
   → Different model TYPES, same full data, combine predictions.
   → Reduces error mainly by diversifying model *families*.

BaggingClassifier/Regressor
   → Same model TYPE, different bootstrap SAMPLES of rows (and optionally columns).
   → Reduces VARIANCE of a single high-variance base learner.

RandomForestClassifier/Regressor
   → BaggingClassifier/Regressor, base learner fixed to DecisionTree,
     PLUS random feature subsampling at every individual split (not just per tree).
   → Reduces variance even further than plain bagged trees, by decorrelating the trees.
```

**Practical tuning order for Random Forest** (most to least impactful, typically):
1. `n_estimators` — set high enough that OOB/CV score plateaus (e.g. 200–500).
2. `max_features` — controls the bias/variance trade-off most directly; grid-search `sqrt`, `log2`, and a couple of fractions.
3. `max_depth` / `min_samples_leaf` — regularize if you see overfitting (train score ≫ OOB/test score).
4. `class_weight` — set if classes are imbalanced.
5. `ccp_alpha` — fine-tuning pruning once the above are reasonable.

Use `oob_score=True` as a cheap first check before setting up full cross-validation — if OOB score and CV score disagree wildly, something else (data leakage, non-i.i.d. splits) is likely going on.
