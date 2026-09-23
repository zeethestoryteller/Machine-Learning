# Simple Guide to Feature Selection

## What is feature selection and why bother?

Not every feature in your dataset actually helps your model make better predictions. Some features are just noise. Removing the useless ones:
- Makes your dataset **smaller** (faster to train on)
- Can **improve model performance** by removing noise
- Reduces **computation cost**

Sklearn gives you tools for this in `sklearn.feature_selection`, split into two main families:

| Type | Idea in plain words |
|---|---|
| **Filter-based** | Look at each feature on its own (using stats), pick the "good" ones — doesn't care what model you'll use |
| **Wrapper-based** | Actually train a model and see which features it thinks are important |

---

## Part 1: Filter-based methods

These just look at the data statistically — fast, but they don't know anything about your actual model.

### 1. VarianceThreshold — "remove boring features"

If a feature barely changes across all your samples (low variance), it's probably not giving your model any useful signal. This removes such features.

- By default, it removes features that have the **exact same value everywhere** (zero variance) — literally useless features.
- You can set a higher threshold to be more aggressive.

```python
from sklearn.feature_selection import VarianceThreshold

vt = VarianceThreshold(threshold=0.01)
X_new = vt.fit_transform(X)
```

---

### 2. Univariate feature selection — "score each feature, keep the best"

These methods score every feature individually using a statistical test, then keep the top ones.

**Three APIs, same idea, different "how many to keep":**

| API | Keeps |
|---|---|
| `SelectKBest` | the top **k** features |
| `SelectPercentile` | the top **X%** of features |
| `GenericUnivariateSelect` | flexible — lets you pick the selection strategy (k_best, percentile, fpr, fdr, fwe) |

```python
from sklearn.feature_selection import SelectKBest, chi2

skb = SelectKBest(chi2, k=20)
X_new = skb.fit_transform(X, y)
```

```python
from sklearn.feature_selection import SelectPercentile, chi2

sp = SelectPercentile(chi2, percentile=20)
X_new = sp.fit_transform(X, y)
```

```python
from sklearn.feature_selection import GenericUnivariateSelect, chi2

transformer = GenericUnivariateSelect(chi2, mode='k_best', param=20)
X_new = transformer.fit_transform(X, y)
```

There are also 3 more specialized filters based on statistical error rates: `SelectFpr` (false positive rate), `SelectFdr` (false discovery rate), `SelectFwe` (family-wise error rate) — these are more advanced statistical variants, useful if you want tighter control over false positives when testing many features at once.

### What "scoring function" should you use?

Every univariate method above needs a scoring function to judge each feature. Three choices:

| Scoring function | Works for | What it measures |
|---|---|---|
| `mutual_info_classif` / `mutual_info_regression` | classification / regression | How dependent two variables are (0 = independent, higher = more dependent) |
| `f_classif` / `f_regression` | classification / regression | F-statistics based scoring |
| `chi2` | **classification only** | How correlated a non-negative feature is with the class label |

⚠️ **Important rule:** Don't use a regression scoring function on a classification problem (or vice versa) — you'll get useless results.

💡 Mutual information and chi-square are especially recommended for **sparse data**.

---

## Part 2: Wrapper-based methods

Unlike filter methods, these actually **use a real model (estimator)** to figure out which features matter — generally more accurate, but slower since they involve training the model multiple times.

### 3. RFE (Recursive Feature Elimination) — "train, remove the weakest, repeat"

How it works:
1. Train a model on **all** features
2. Look at feature importance from the model
3. **Remove the least important feature**
4. Repeat until you're down to your desired number of features

```python
from sklearn.feature_selection import RFE
from sklearn.svm import SVC

rfe = RFE(estimator=SVC(kernel="linear"), n_features_to_select=10)
X_new = rfe.fit_transform(X, y)
```

**Don't know how many features you want?** Use `RFECV` instead — it runs RFE inside a cross-validation loop to automatically find the best number of features.

```python
from sklearn.feature_selection import RFECV

rfecv = RFECV(estimator=SVC(kernel="linear"))
X_new = rfecv.fit_transform(X, y)
```

---

### 4. SelectFromModel — "keep features above an importance threshold"

Trains a model once, then keeps only features whose importance is above a certain threshold (or keeps a fixed max number of top features).

- Gets feature importance from `coef_`, `feature_importances_`, or a custom `importance_getter`
- Threshold can be a number, or a smart string like `'mean'`, `'median'`, or `'0.1*mean'`

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC

clf = LinearSVC(C=0.01, penalty="l1", dual=False)
clf = clf.fit(X, y)

model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X)
```
This example keeps only the features that ended up with **non-zero weights** after training an L1-regularized linear SVM (L1 regularization naturally zeroes out unimportant features).

---

### 5. SequentialFeatureSelector — "add or remove features one at a time, greedily"

Two strategies:

| Strategy | How it works |
|---|---|
| **Forward selection** | Start with 0 features. Add one feature at a time — whichever one improves cross-validation score the most. Repeat. |
| **Backward selection** | Start with **all** features. Remove the least useful one at a time. Repeat. |

```python
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier

sfs = SequentialFeatureSelector(
    KNeighborsClassifier(),
    n_features_to_select=5,
    direction='backward'  # or 'forward'
)
X_new = sfs.fit_transform(X, y)
```

**Which direction should you pick?** Pick whichever needs fewer iterations for your target number of features:
- Want to select **7 out of 10** features? → **Backward** is faster (only 3 steps to remove, vs. 7 steps to add)
- Want to select **2 out of 10** features? → **Forward** is faster

### Trade-offs of Sequential Feature Selection
- ✅ Doesn't require the model to have `coef_` or `feature_importances_` (unlike RFE/SelectFromModel) — works with **any** estimator
- ❌ Much **slower** — it has to train many models. For backward selection going from *m* features to *m−1* using *k*-fold CV, it needs to fit `m × k` models — compare that to RFE (1 fit) or SelectFromModel (1 fit, no iteration at all)

---

## Simple summary: which one should I use?

| I want to... | Use |
|---|---|
| Quickly drop useless/constant features | `VarianceThreshold` |
| Score features statistically and keep top-k or top-% | `SelectKBest` / `SelectPercentile` |
| Let a trained model tell me which features matter, removing one at a time | `RFE` (or `RFECV` if unsure how many to keep) |
| Let a trained model tell me which features matter, in one shot | `SelectFromModel` |
| Try any model, don't need coef_/feature_importances_, willing to wait | `SequentialFeatureSelector` |

**General rule of thumb:** Filter methods (Part 1) are fast — use them for quick, cheap pruning, especially with huge datasets. Wrapper methods (Part 2) are more accurate but slower — use them when you can afford the extra compute and want performance-tuned feature selection.
