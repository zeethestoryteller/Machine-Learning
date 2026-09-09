# K-Nearest Neighbors (KNN) — Complete Guide

## 1. What is KNN, really?

KNN belongs to a family called **instance-based** (or **non-generalizing**) learning. That name tells you everything about how it behaves:

- It does **not** build a model during training (no equation, no weights being learned).
- It just **memorizes** the training data.
- At prediction time, it looks at the new point, finds its "neighbors" in the stored training data, and lets them **vote** on the label (classification) or **average** their values (regression).

**Why this matters:** because there's no real "training" step, KNN's `fit()` is nearly instant — it's just storing data. But prediction can be slow, because for every new point it has to search through the stored data to find neighbors. This is the central trade-off of KNN: cheap training, expensive prediction.

scikit-learn gives you two flavors:

| | **KNeighborsClassifier** | **RadiusNeighborsClassifier** |
|---|---|---|
| Neighbor rule | the **k** closest points | **all** points within a fixed radius `r` |
| When to use | default choice, most common | when data density varies a lot across the space |
| Behavior | always uses exactly k neighbors | dense regions use more neighbors, sparse regions use fewer |

**Why the radius version exists:** imagine your data isn't spread evenly — some regions are packed with points, others are sparse. With a fixed `k`, a point in a sparse region might have its "5 nearest neighbors" scattered far away, giving a misleading vote. RadiusNeighborsClassifier fixes the *distance* instead, so a sparse-region point naturally gets fewer (but genuinely close) neighbors, and a dense-region point gets more.

---

## 2. Basic usage — the two-step pattern

Every scikit-learn estimator follows this pattern, and KNN is no exception:

```python
# Step 1: create the estimator object
from sklearn.neighbors import KNeighborsClassifier
kneighbor_classifier = KNeighborsClassifier()

# Step 2: fit it on your training data
# X_train = feature matrix, y_train = label vector
kneighbor_classifier.fit(X_train, y_train)

# Step 3 (not shown in the slides, but essential): predict
y_pred = kneighbor_classifier.predict(X_test)
```

Same pattern for the radius version:

```python
from sklearn.neighbors import RadiusNeighborsClassifier
radius_classifier = RadiusNeighborsClassifier()
radius_classifier.fit(X_train, y_train)
y_pred = radius_classifier.predict(X_test)
```

---

## 3. Parameters — what each one does and *why* it exists

### `n_neighbors` (KNeighborsClassifier only)

```python
kneighbor_classifier = KNeighborsClassifier(n_neighbors=3)
```

- **What it is:** the value of *k* — how many nearest points to look at.
- **Default:** `5`
- **Why it matters:** this is the single most important hyperparameter in KNN.
  - **Too small k** (e.g. k=1): the model reacts to every tiny fluctuation in the data — it overfits, is very sensitive to noise/outliers, and produces a jagged, unstable decision boundary.
  - **Too large k**: predictions get smoothed over too broad a neighborhood, and you start "hearing" the vote of points that aren't really representative of the local pattern — the model underfits and the decision boundary becomes too simple. In the extreme, k = size of training set just predicts the majority class everywhere.
  - **Rule of thumb:** k should be tuned via cross-validation. Odd values of k are often preferred in binary classification to avoid tie votes.

### `weights`

```python
kneighbor_classifier = KNeighborsClassifier(weights='uniform')  # default
```

- **What it is:** controls how much "say" each neighbor gets in the vote.
- **Options:**
  - `'uniform'` (default): every one of the k neighbors gets an equal vote, regardless of whether it's the closest or the farthest of the k.
  - `'distance'`: neighbors are weighted by the **inverse of their distance** — so a neighbor right next to the query point influences the prediction much more than one that's barely within the k-nearest cutoff.
  - **A custom callable**: you can pass your own function.
- **Why it matters:** with `'uniform'`, a neighbor that's barely inside the k-cutoff counts exactly as much as one sitting almost on top of the query point — that can be misleading. `'distance'` fixes this by trusting closer points more, which often improves accuracy, especially when k is larger.

**Custom weight function example:**
```python
def user_weights(weights_array):
    # weights_array = array of distances to each neighbor
    return weights_array  # return an array of the same shape = the weights

kneighbor_classifier = KNeighborsClassifier(weights=user_weights)
```
This is useful when neither uniform nor simple inverse-distance weighting fits your problem — e.g., you might want to weight by inverse-squared distance, or apply a domain-specific decay function.

### `algorithm`

```python
kneighbor_classifier = KNeighborsClassifier(algorithm='auto')  # default
```

- **What it is:** the internal data structure/search strategy used to actually *find* the nearest neighbors.
- **Options:**
  - `'brute'`: brute-force — computes distance to every single training point. Simple, guaranteed correct, but slow for large datasets (O(n) per query).
  - `'kd_tree'`: builds a KDTree, which partitions space to avoid comparing against every point. Fast in low-to-moderate dimensions.
  - `'ball_tree'`: builds a BallTree, which handles higher-dimensional data and more exotic distance metrics better than KDTree.
  - `'auto'`: sklearn looks at your data (size, dimensionality, sparsity) at `fit()` time and picks whichever of the above is likely fastest.
- **Why it matters:** this parameter is purely about **speed**, not accuracy — all four give the same neighbors, just found differently. KD-trees and Ball-trees become less effective as dimensionality grows (the "curse of dimensionality"), at which point brute-force can actually become competitive again. Letting sklearn decide (`'auto'`) is usually the right call unless you have a specific reason to override it.

### `leaf_size` (only relevant for `'ball_tree'` / `'kd_tree'`)

```python
kneighbor_classifier = KNeighborsClassifier(algorithm='kd_tree', leaf_size=30)  # default
```

- **What it is:** the size of the "leaf" node at which the tree stops subdividing and switches to brute-force search within that leaf.
- **Default:** `30`
- **Why it matters:** it's a speed/memory trade-off knob.
  - Smaller leaf_size → deeper tree → more overhead building/traversing the tree, but faster individual leaf-level search.
  - Larger leaf_size → shallower tree → less construction overhead, but more brute-force comparisons per leaf.
  - It does **not** change your results — only how fast you get them. Most people leave it at the default unless profiling shows a bottleneck.

### `metric` and `p` — how "distance" is defined

```python
kneighbor_classifier = KNeighborsClassifier(metric='minkowski', p=2)  # both are defaults
```

- **`metric`:** the distance function used to decide who counts as a "neighbor." Options include `"euclidean"`, `"manhattan"`, `"chebyshev"`, `"minkowski"`, `"seuclidean"`, `"mahalanobis"`, or your own callable.
- **`p`:** only used when `metric='minkowski'` — it's the power parameter of the Minkowski distance formula:
  $$ d(x,y) = \left(\sum_i |x_i - y_i|^p\right)^{1/p} $$
  - `p=1` → Manhattan distance (sum of absolute differences)
  - `p=2` → Euclidean distance (straight-line distance) — **this is the default combination** (`metric='minkowski', p=2` ≡ Euclidean)
- **Why it matters:** this defines what "close" even *means* for your data, which is arguably as important as k itself.
  - Euclidean distance assumes all features are on comparable, continuous, roughly-symmetric scales — that's why **feature scaling (standardization) before KNN is essential**, otherwise a feature measured in the thousands (like income) will dominate a feature measured in single digits (like age).
  - Manhattan distance is more robust to outliers and is often preferred in high-dimensional or grid-like data.
  - Mahalanobis distance accounts for correlations between features — useful when your features aren't independent.

### `radius` (RadiusNeighborsClassifier only)

```python
radius_classifier = RadiusNeighborsClassifier(radius=1.0)  # default
```

- **What it is:** the fixed distance `r` — any training point within this distance of the query point counts as a neighbor and gets a vote.
- **Default:** `1.0`
- **Why it matters:** this is the RadiusNeighborsClassifier equivalent of tuning `k` — it directly controls how "local" the vote is. Too small a radius and some query points may have **zero** neighbors (a real failure mode you need to handle); too large and you dilute the vote with irrelevant, far-away points. Like `k`, it should be scaled relative to your (scaled) feature space and tuned via cross-validation.

RadiusNeighborsClassifier shares `weights`, `algorithm`, `leaf_size`, `metric`, and `p` with the same meanings described above.

---

## 4. Putting it all together — a full worked example

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load data
X, y = load_iris(return_X_y=True)

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Scale features — CRITICAL for KNN since it relies on distances
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # use the SAME scaler, don't refit

# 4. Instantiate and fit
knn = KNeighborsClassifier(
    n_neighbors=5,        # how many neighbors vote
    weights='distance',   # closer neighbors count more
    algorithm='auto',     # let sklearn pick the fastest search method
    metric='minkowski',
    p=2                    # Euclidean distance
)
knn.fit(X_train_scaled, y_train)

# 5. Predict and evaluate
y_pred = knn.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 6. Tune k with cross-validation instead of guessing
param_grid = {'n_neighbors': range(1, 21), 'weights': ['uniform', 'distance']}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid.fit(X_train_scaled, y_train)
print("Best k / weights:", grid.best_params_)
print("Best CV accuracy:", grid.best_score_)
```

**Why each step is there:**
- **Scaling (step 3)** — without it, features with larger numeric ranges silently dominate the distance calculation.
- **Fit scaler on train only** — fitting on test data would leak information from the test set into preprocessing (data leakage).
- **GridSearchCV (step 6)** — since `k` has no "correct" theoretical value, the standard practice is to try a range of values and pick whichever generalizes best on held-out folds, rather than guessing.

---

## 5. Quick mental model to remember it all

- **Training = "just remember everything."** No real learning happens at `fit()`.
- **Prediction = "look around and vote."** `k` (or `radius`) decides *how far* to look; `weights` decides *how loudly* each neighbor's vote counts; `metric`/`p` decide *what "close" means*; `algorithm`/`leaf_size` decide *how fast* the search runs (no effect on the answer).
- **The one non-negotiable prep step:** scale your features before using KNN, since it's 100% distance-based.

---

*Source: "K Nearest Neighbours" — Dr. Ashish Tendulkar, IIT Madras, Machine Learning Practice.*
