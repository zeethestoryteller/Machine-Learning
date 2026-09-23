# Guide to Feature Transformers


## Part A: Feature Extraction (`sklearn.feature_extraction`)

### 1. DictVectorizer

Converts a list of dictionaries (feature name → feature value mappings) into a numeric matrix.

**Example:**
```python
data = [
    {'age': 4, 'height': 96.0},
    {'age': 1, 'height': 73.9},
    {'age': 3, 'height': 88.9},
    {'age': 2, 'height': 81.6}
]
```

```python
from sklearn.feature_extraction import DictVectorizer

dv = DictVectorizer(sparse=False)
X_transformed = dv.fit_transform(data)
```
Result:
```
[[4, 96.0],
 [1, 73.9],
 [3, 88.9],
 [2, 81.6]]
```

**Use when:** your data naturally comes as a list of feature dicts (common with JSON-like records).

---

### 2. FeatureHasher

A **high-speed, low-memory** vectorizer using the *feature hashing* technique. Instead of maintaining a hash table of feature names (like DictVectorizer), it directly applies a hash function to determine column index.

**Key characteristics:**
- Faster and more memory-efficient than DictVectorizer
- **Trade-off:** loses inspectability — it doesn't remember original feature names and has **no `inverse_transform`**
- Output is a `scipy.sparse` matrix

```python
from sklearn.feature_extraction import FeatureHasher

fh = FeatureHasher(n_features=10)
X_transformed = fh.fit_transform(data)  # data as list of dicts
```

**Use when:** you have very high-dimensional/high-cardinality features (e.g., large vocabularies) and need speed/memory efficiency over interpretability.

### DictVectorizer vs FeatureHasher

| | DictVectorizer | FeatureHasher |
|---|---|---|
| Speed/memory | Slower, more memory | Faster, low memory |
| Inspectable (inverse_transform) | Yes | No |
| Output | Dense or sparse | Always sparse |

Sklearn also provides dedicated APIs for **images** (`sklearn.feature_extraction.image`) and **text** (`sklearn.feature_extraction.text`) — see the sklearn user guide for details on those.

---

## Part B: Numeric Transformers

### 3. FunctionTransformer

Applies a **user-defined function** to construct transformed features — useful for custom math transforms like log-scaling.

**Example:**
```
X = [[128, 2],
     [2,   256],
     [4,   1],
     [512, 64]]
```

```python
import numpy
from sklearn.preprocessing import FunctionTransformer

ft = FunctionTransformer(numpy.log2)
X_transformed = ft.fit_transform(X)
```
Result:
```
[[7, 1],
 [1, 8],
 [2, 0],
 [9, 6]]
```

**Use when:** you need a custom, arbitrary transformation (log, sqrt, reciprocal, or any callable) applied consistently within a pipeline.

---

### 4. PolynomialFeatures

Generates a new feature matrix consisting of **all polynomial combinations** of features up to a specified degree.

**Example:** Given `X = [x1, x2]`

```python
from sklearn.preprocessing import PolynomialFeatures

pf = PolynomialFeatures(degree=2)
X_transformed = pf.fit_transform(X)
# Result: [x1, x2, x1*x2, x1², x2²]
```

```python
pf = PolynomialFeatures(degree=3)
X_transformed = pf.fit_transform(X)
# Result: [x1, x2, x1*x2, x1², x2², x1²*x2, x1*x2², x1³, x2³]
```

**Use when:** you want to capture non-linear relationships/interactions between features for use in a linear model (this is how you build polynomial regression on top of `LinearRegression`).

⚠️ **Caution:** degree and feature count grow combinatorially — with many original features, higher degrees quickly explode the feature space.

---

### 5. KBinsDiscretizer

Divides a **continuous variable into bins**, then applies one-hot or ordinal encoding to the resulting bin labels.

**Example:**
```
x = [0, 0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0]
```

```python
from sklearn.preprocessing import KBinsDiscretizer

kbd = KBinsDiscretizer(n_bins=5, strategy='uniform', encode='ordinal')
x_transformed = kbd.fit_transform(x.reshape(-1, 1))
```
Result: `[0., 0., 1., 1., 2., 3., 3., 4., 4.]`

**Key parameters:**
| Parameter | Options |
|---|---|
| `n_bins` | number of bins |
| `strategy` | `'uniform'` (equal-width bins), `'quantile'`, `'kmeans'` |
| `encode` | `'ordinal'`, `'onehot'`, `'onehot-dense'` |

**Use when:** you want to convert a continuous feature into categorical buckets — useful for tree-based models, or to help linear models capture non-linear/threshold effects.

---

### 6. add_dummy_feature

Augments the dataset with a **column of all 1s** (a bias/intercept column).

**Example:**
```
X = [[7, 1],
     [1, 8],
     [2, 0],
     [9, 6]]
```

```python
from sklearn.preprocessing import add_dummy_feature

X_transformed = add_dummy_feature(X)
```
Result:
```
[[1, 7, 1],
 [1, 1, 8],
 [1, 2, 0],
 [1, 9, 6]]
```

**Use when:** an estimator needs an explicit intercept/bias term as part of the feature matrix rather than fitting one internally.

---

## Quick summary table

| Transformer | Purpose | Module |
|---|---|---|
| DictVectorizer | dict records → numeric matrix | `sklearn.feature_extraction` |
| FeatureHasher | fast/low-memory hashing-based vectorization | `sklearn.feature_extraction` |
| FunctionTransformer | apply custom function (e.g., log) | `sklearn.preprocessing` |
| PolynomialFeatures | generate polynomial/interaction terms | `sklearn.preprocessing` |
| KBinsDiscretizer | bin continuous → categorical | `sklearn.preprocessing` |
| add_dummy_feature | add bias/intercept column | `sklearn.preprocessing` |

---

## Best practice: combine in a pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression

pipe = Pipeline(steps=[
    ('poly', PolynomialFeatures(degree=2)),
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

pipe.fit(X_train, y_train)
predictions = pipe.predict(X_test)
```
