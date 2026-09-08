# Machine Learning Practice — Study Guide
*Covers every concept needed to solve the 60-question practice exam*

---

## 1. Feature Scaling

### StandardScaler
Transforms data so mean = 0, std = 1 (using **population std, ddof=0**):
```
z = (x - mean) / std
```
- `fit(X)` computes mean and std **from the training data**.
- `transform(X_new)` applies the *stored* mean/std — it does NOT recompute them.

**Worked pattern (Q6 style):** X = [2,6,10,14] → mean = 8, std = sqrt(((2-8)²+(6-8)²+(10-8)²+(14-8)²)/4) = sqrt((36+4+4+36)/4) = sqrt(20) ≈ 4.472
For X_new = [8]: z = (8-8)/4.472 = **0**

### MinMaxScaler
```
x_scaled = (x - min) / (max - min)   [default range 0–1]
```
With `feature_range=(a,b)`: `x_scaled = a + (x-min)/(max-min) * (b-a)`
- Again, `fit` learns min/max from training data; `transform` reuses them, even for new out-of-range values (result can exceed [0,1] if new value is outside original range).

**Key exam trap:** scaler is *fit* on one array, then *transform* is called on a *different* array (`X_new`). Always use the min/max/mean/std from the **fit** data, not the new data.

### Which algorithms are affected by scaling?
| Affected (distance/gradient based) | Not affected (split/rule based, or scale-invariant) |
|---|---|
| KNN, SVM, K-Means, Linear/Logistic Regression (gradient descent), Neural Nets | Decision Tree, Random Forest, Naive Bayes* |

*Naive Bayes and tree-based models split/multiply probabilities in ways that don't depend on feature magnitude. LinearRegression's closed-form solution is technically scale-invariant in outcome, but gradient-based fitting (like SGD) is affected — sklearn's default LinearRegression uses least squares (not affected in coefficients' predictive power, but coefficients scale). For exam purposes: **KNN, KMeans, SVM → affected; DecisionTree, NaiveBayes → not affected**.

---

## 2. Encoding Categorical Data

### LabelEncoder
Assigns integers alphabetically to unique categories.
```python
words = ['apple','banana','cherry','banana','cherry','banana','cherry','apple','apple','banana','banana']
```
Alphabetical order → apple=0, banana=1, cherry=2. Count how many times 'apple' appears to answer "how many zeros."

### OneHotEncoder
Creates a binary column per category.
- `sparse_output=False` → dense numpy array.
- **Sparsity %** = (number of 0s) / (total cells) × 100.
  - 5 words, all unique categories → 5×5 identity-like matrix → 20 zeros / 25 cells = 80%.
- `drop='first'` drops one category column to avoid multicollinearity (used with ColumnTransformer).

### MultiLabelBinarizer
Used for **multi-label** data (each sample can belong to several classes at once), e.g. `['large','cold']`.
```python
mlb = MultiLabelBinarizer(classes=['cold','hot','large','small'])
```
Output columns follow the **order given in `classes=`**, and each row gets a 1 in the position of every label present in that sample (order-independent, treats each row as a *set* of labels, output length = len(classes)).
- `['large','cold']` → cold=1, hot=0, large=1, small=0 → `[1,0,1,0]`

---

## 3. ColumnTransformer & Pipelines

### ColumnTransformer
Combines multiple preprocessing steps, applied to specified columns, and **concatenates outputs column-wise**.
```python
ct = ColumnTransformer([
    ('num', StandardScaler(), ['Height','Weight']),      # 2 cols → 2 cols
    ('cat', OneHotEncoder(drop='first'), ['City'])        # depends on unique categories
])
```
**Shape logic:**
- Numeric columns pass through unchanged in count (2 → 2).
- OneHotEncoder without `drop`: k categories → k columns.
- OneHotEncoder **with `drop='first'`**: k categories → (k−1) columns.

Example: City has 3 unique values (A, B, C) → OHE with drop='first' → 2 columns.
Total columns = 2 (numeric) + 2 (OHE) = **4**. Rows unchanged (4 rows in, 4 rows out) → shape **(4,4)**.

If City has only 2 unique values (e.g., 'M','F') → drop='first' → 1 column.
Total = 2 + 1 = 3 columns, 3 rows → shape **(3,3)**.

### Pipeline
Chains steps; each step's `fit_transform` output feeds the next step's input. Final step is usually an estimator (`.fit()`), so you call `pipeline.fit(X,y)` — **not** `fit_transform` when the last step is a model (fit_transform would still "work" if all steps support transform, but conventionally use `.fit()` then `.predict()`).

**Common bug (Q31 style):** calling `model.fit_transform(X, y)` when the last pipeline step is `LinearRegression` (which has no `transform`) → this actually raises an AttributeError. The fix: use `model.fit(X, y)`.

### FeatureUnion
Runs multiple transformers **in parallel** on the same input and concatenates their outputs (unlike Pipeline which chains sequentially).
```python
combined = FeatureUnion([('poly', PolynomialFeatures(degree=3, include_bias=False)),
                          ('pca', PCA(n_components=2))])
```
Total output features = (features from poly) + (features from pca).
- PolynomialFeatures with 4 input features, degree=3, include_bias=False: total polynomial terms = C(n+d, d) − 1 (excluding bias term). For n=4, d=3: total combos with bias = C(7,3)=35, minus bias(1) = 34 features.
- PCA(n_components=2) → 2 features.
- Total = 34 + 2 = 36 → shape (150, 36).

**General PolynomialFeatures formula:** number of output features (degree d, n input features, include_bias=True) = C(n+d, d). Subtract 1 if `include_bias=False`.

---

## 4. Missing Value Imputation

### SimpleImputer
```python
SimpleImputer(strategy='mean')   # or 'median', 'most_frequent'
```
Replaces NaN in each column with that column's mean/median (computed ignoring NaNs), **column-wise independently**.

**Worked pattern (Q15):**
```
F3 column: 9, 7, ?, 5   → mean of (9,7,5) = 21/3 = 7
```
Missing value = **7**.

### KNNImputer
```python
KNNImputer(n_neighbors=2)
```
Fills missing value using the **average of that feature's value from the k nearest neighbor rows** (distance computed using available features, e.g., F1, F2), not just the column mean.

**Approach:** Compute distance between the row with the missing value and all other rows using the *complete* features (F1, F2). Take the k=2 nearest rows, average their F3 values.

### Group-wise imputation with pandas
```python
df["Age"] = df.groupby("Department")["Age"].transform(lambda x: x.fillna(x.mean()))
```
- Use `.transform()` (not `.apply()`) when you need the output to have the **same shape/index as the original column** so it can be assigned back directly. `apply` with a lambda returning a Series *can* also work here in newer pandas but `transform` is the canonical/safe choice for this pattern.
- Salary → simple column-wide median: `df["Salary"].fillna(df["Salary"].median())`.

---

## 5. Supervised Learning — Classification

### KNeighborsClassifier
Predicts the class by **majority vote of the k nearest neighbors** (by Euclidean distance, unless specified).

**Worked pattern (Q16):**
```
X = [[2,2],[3,3],[5,5],[12,12],[13,13]], y=[0,0,0,1,1]
knn = KNeighborsClassifier(n_neighbors=3)
test_point = [[6,6]]
```
Distances from (6,6): to (2,2)=5.66, (3,3)=4.24, (5,5)=1.41, (12,12)=8.49, (13,13)=9.90
3 nearest: (5,5)→y=0, (3,3)→y=0, (2,2)→y=0 → majority = **0**

### DecisionTreeClassifier — split conditions
A split at node N happens only if **both**:
1. N has ≥ `min_samples_split` samples.
2. **Both** children have ≥ `min_samples_leaf` samples.

Check each option against both rules. E.g., "Node has 6 samples, split into 4/2" with `min_samples_split=6, min_samples_leaf=4` → node has exactly 6 (✓ rule 1) but right child has only 2 < 4 (✗ rule 2) → split NOT allowed.

### RandomForestClassifier / RandomForestRegressor — key parameters
| Parameter | Meaning |
|---|---|
| `n_estimators` | Number of trees in the forest |
| `max_depth` | Max depth of **each individual tree** |
| `min_samples_leaf` | Min samples required in a leaf node (not "number of trees") |
| `bootstrap` | Whether **each tree** is trained on a bootstrap (random sample with replacement) of the data |
| `max_features` | Number of features **each tree considers at each split** (not "number of trees using subsets") |

Key concept: `max_features=5` means **every tree**, at **every split**, randomly considers only 5 features (not "5 out of 200 trees"). `bootstrap=True` means each tree gets its own bootstrap sample (not the "same shuffled full dataset").

### BaggingClassifier
```python
BaggingClassifier(base_estimator, n_estimators=30, bootstrap=False, ...)
```
- `bootstrap=False` → samples drawn **without replacement** (so it's NOT bootstrapping — pasting instead).
- `weights="distance"` in the base KNN → closer neighbors get **more voting weight** (not equal).
- Ensemble consists of `n_estimators` copies of the base estimator (e.g., 30, not "3").
- With `bootstrap=False`, there's no formal "out-of-bag" sample (OOB requires bootstrap=True).

### VotingClassifier
- `voting='hard'` → majority vote of predicted **class labels**.
- `voting='soft'` → **averages predicted class probabilities**, picks the class with highest average probability. (Requires all base estimators support `predict_proba`, hence `SVC(probability=True)`.)

### Perceptron / SGD-based models — warm_start
By default, calling `.fit()` again **resets** the model and retrains from scratch.
- Set `warm_start=True` to make subsequent `.fit()` calls **continue training** from the previously learned weights instead of resetting.

---

## 6. Model Evaluation Metrics

### Confusion-matrix metrics
```
                Predicted 1    Predicted 0
Actual 1        TP             FN
Actual 0        FP             TN
```
- **Precision** = TP / (TP + FP) — "of predicted positives, how many correct"
- **Recall** = TP / (TP + FN) — "of actual positives, how many found"
- **F1** = 2·(Precision·Recall)/(Precision+Recall)
- **Accuracy** = (TP+TN)/Total

**Worked pattern (Q7):**
```
y_true=[1,0,1,1,0,1], y_pred=[1,0,0,1,0,1]
```
TP (pred=1,true=1): positions 0,3,5 → 3
FN (pred=0,true=1): position 2 → 1
Recall = 3/(3+1) = **0.75**

**Reading custom metric functions:** Given code like:
```python
m = (y_pred==1) & (y_true==1)   # TP
n = (y_pred==0) & (y_true==1)   # FN
return a/(a+b)  # = TP/(TP+FN) = Recall
```
If instead `n = (y_pred==1) & (y_true==0)` (FP), then `a/(a+b)` = TP/(TP+FP) = **Precision**.
Always identify what `m` and `n` represent (TP, FP, FN, TN) then match to the formula.

### Clustering evaluation
- **Silhouette Score** — measures how well-separated clusters are (higher = better), works without ground truth.
- **Inertia** — sum of squared distances of samples to nearest cluster center (K-Means specific, `.inertia_` attribute).
- Metrics like F1, R2, Negative MAE — **not** used for unsupervised clustering (they need ground-truth labels or continuous targets).

### classification_report
Includes: **Accuracy, Precision, Recall, F1-score** (per class + averages). Does **NOT** include MSE or Cross Entropy Loss (those are regression/loss metrics, not classification report contents).

### Cosine Similarity
```
cos_sim(A,B) = (A · B) / (||A|| * ||B||)
```
Correct implementation:
```python
def cos_sim(a, b):
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
```
(`a @ b` or `np.dot(a,b)` = dot product; `np.linalg.norm` = magnitude. Options using `np.sum(a*b)/(np.sum(a)*np.sum(b))` or `a+b` in numerator are wrong.)

**Worked pattern (Q11):** A=(1,2,3), B=(1,0,0)
Dot = 1×1+2×0+3×0 = 1
‖A‖ = √14 ≈ 3.742, ‖B‖ = 1
cos_sim = 1/3.742 ≈ **0.27**

**Worked pattern (Q55):** A=(1,2,3), B=(2,0,1)
Dot = 2+0+3 = 5
‖A‖=√14≈3.742, ‖B‖=√5≈2.236
cos_sim = 5/(3.742×2.236) ≈ 5/8.367 ≈ **0.60**

---

## 7. Train/Test Split Behavior

```python
train_test_split(X, y, test_size=0.2, random_state=42)
```
- Same `random_state` → **same** row split every time (reproducible).
- Different `random_state` → **different** split (different rows selected), even with the same `test_size`.
- `train_size=0.8` is mathematically equivalent to `test_size=0.2`, but if `random_state` also matches, the **split will be identical** (same rows) — only the labeling of "train_size vs test_size" differs, not the algorithm.

So: Case1 (test_size=0.2, random_state=42) vs Case3 (train_size=0.8, random_state=42) → **same random_state and equivalent proportions → same split**.
Case1 vs Case2 (different random_state) → **different split**.

---

## 8. Hyperparameter Tuning — GridSearchCV

Total combinations = **product of the number of values for each hyperparameter** (Cartesian product), NOT the sum.

**Worked pattern (Q21):**
```python
param_grid = {
    "n_estimators": [50,100,150,200],     # 4 values
    "max_depth": [None,5,10,15,20],       # 5 values
    "min_samples_split": [2,3,4,5,6,8,10] # 7 values
}
```
Total = 4 × 5 × 7 = **140**

**Q39-style:** A has 3 values, B has 4 values → combinations = 3×4 = **12** (not 7 — that's the sum, which is wrong).

- `best_score_` = the **mean** cross-validated score (averaged across folds) of the best parameter combination — not the max fold score.
- `scoring` parameter accepts many options beyond R² and MSE (e.g., `neg_mean_squared_error`, `neg_mean_absolute_error`, custom scorers) — the claim "no other scoring metric can be used" is **false**.
- Pipeline step naming in param_grid: use `stepname__parameter` (double underscore), e.g., `'poly__degree': [2,3,...]` when the pipeline step is named `'poly'`.

---

## 9. Clustering

### K-Means
```python
kmeans = KMeans(n_clusters=3)
kmeans.fit(data)
labels = kmeans.labels_
```
- `.labels_` → the **cluster index (0,1,2,...) assigned to each data point** (not centroids, not feature values, not cluster count).
- `.cluster_centers_` → coordinates of centroids (different attribute).
- `.inertia_` → sum of squared distances to nearest centroid.

### make_blobs
```python
make_blobs(n_samples=8, centers=2, n_features=2, cluster_std=0.5, random_state=42)
```
- Creates `n_samples` points around `centers` cluster centers.
- Cluster sizes are **not required to be exactly equal** — sklearn distributes samples roughly evenly but **does not guarantee exact equal split** (8 samples / 2 centers won't error, and won't necessarily be exactly 4/4 either, though with random assignment it often is close). The safest general answer: "two clusters, not necessarily equal-sized; distribution may vary."
- It does **not** raise an error just because `n_samples` isn't perfectly divisible by `centers`.

### Hierarchical Agglomerative Clustering
- **Bottom-up approach**: starts with each point as its own cluster, and **repeatedly merges the two closest clusters** until a stopping criterion (e.g., desired number of clusters) is met.
- Does NOT require specifying number of clusters *before* training (unlike K-Means) — you can cut the dendrogram afterward.
- Does NOT update centroids (that's K-Means); it uses linkage criteria (single/complete/average/ward) to decide which clusters are "closest."

---

## 10. Recommendation Systems

| Method | Needs | Handles new/no-rating items? |
|---|---|---|
| **Collaborative filtering** (user-based or item-based) | User-item interaction/rating matrix | ❌ Cold-start problem — can't recommend unrated items |
| **Content-based filtering** | Item metadata (genre, cast, director, etc.) | ✅ Yes — works from item features regardless of ratings |
| **Matrix factorization / SVD** | Interaction matrix | Aims to **reduce dimensionality** of the sparse user-item matrix, revealing latent factors |

- A brand-new movie with no ratings but known metadata → only **content-based filtering** can recommend it.
- SVD on a user-item matrix → purpose is **dimensionality reduction** (finds latent factors), not to "increase sparsity" or "delete history."
- User-based collaborative filtering needs the **user-item interaction matrix** (to find similar users based on shared rating patterns) — not item content or user demographic features.

---

## 11. Time Series (statsmodels)

### Augmented Dickey-Fuller Test — `adfuller()`
Tests whether a time series is **stationary** (constant mean/variance over time, no trend).
```python
stat, pval, n_lags, crit = adfuller(series)
```
- **Null hypothesis (H0):** series is non-stationary.
- If `p-value < 0.05` (significance level) → **reject H0** → series **is stationary**.
- If `p-value ≥ 0.05` → fail to reject H0 → series is **non-stationary**.
- p=0.03 < 0.05 → **series is stationary**.

### ARIMA
```python
model = ARIMA(data, order=(1,1,1))
model_fit = model.fit()
print(model_fit.summary())
```
Correct sequence: instantiate → `.fit()` returns a **results object** → call `.summary()` on the **results object**, not the original model. (`model.summary()` before fitting, or `model.fit()` called without capturing/using the returned results object, are common wrong-option traps.)

### Handling NaNs
`ARIMA(series_with_nans, ...).fit()` — statsmodels does **not** silently auto-impute; it will **raise an error or behave unpredictably depending on the estimator** — the user must handle missing values (e.g., interpolate/impute) beforehand.

---

## 12. Pandas Data Manipulation

```python
result = (
    df[df["Marks"] > 60]
    .sort_values("Age")
    .reset_index(drop=True)
)
```
Read chained pandas operations **left to right**:
1. Filter: keep rows where Marks > 60
2. Sort remaining rows by Age (ascending by default)
3. Reset the index to 0,1,2,... and **drop** the old index (`drop=True` prevents it becoming a new column)

→ "Only rows where Marks > 60, sorted by Age, and index reset."

---

## 13. fit vs fit_transform vs transform — the golden rule

| Situation | Method |
|---|---|
| Learning parameters from **training data** for a transformer (e.g., scaler, encoder) AND applying it | `fit_transform` (on train only) |
| Applying an **already-fitted** transformer to new/test data | `transform` only (never re-fit on test data — this causes data leakage) |
| Training a **model** (not a transformer) | `fit` (models don't have `transform`; use `.predict()` afterward) |

- Fitting a **MinMaxScaler on test data** → **wrong**, should only ever `transform` (using stats learned from train) — never `fit` or `fit_transform` on test data.
- Fitting **MinMaxScaler on train data** → `fit_transform` ✅ correct.
- **Creating a decision tree** (i.e., training the model) → `.fit()` (not `fit_transform` — trees don't transform data).
- **Learning weights of logistic regression** → `.fit()`, not `fit_transform` (LogisticRegression has no `transform` method).

---

## 14. Datatypes — sklearn datasets vs pandas

```python
d1 = load_iris()          # returns a sklearn "Bunch" object (dict-like, with .data, .target, etc.)
d2 = pd.read_csv('file2.csv')   # returns a pandas DataFrame
```
→ **(bunch, dataframe)**

`load_iris(as_frame=True, return_X_y=True)` → returns `(X, y)` where X is a **DataFrame** and y is a **Series** (as_frame changes the return type).

---

## 15. Quick Reference — Common Traps

1. **fit vs transform on new data** — always use stats learned on the *original fit* data, never refit on new/test data.
2. **GridSearchCV combinations = product, not sum** of parameter value counts.
3. **max_features / bootstrap in RandomForest** apply *per tree*, not to a subset of trees.
4. **KMeans `.labels_`** = cluster assignment per point, not centroids.
5. **ADF test:** low p-value (< 0.05) → stationary (reject H0 of non-stationarity).
6. **OneHotEncoder(drop='first')** removes one column per categorical feature to avoid multicollinearity.
7. **Precision vs Recall** — always trace which mask represents FP vs FN in custom metric code.
8. **ColumnTransformer output shape** — sum columns from each transformer; OHE columns = (unique categories − 1) if `drop='first'`.
9. **PolynomialFeatures feature count** = C(n+d, d) − (1 if include_bias=False else 0).
10. **train_test_split** — identical `random_state` (and equivalent split ratio) ⇒ identical rows in train/test, regardless of whether you specify `test_size` or `train_size`.

---

### How to use this guide
For each question: identify which of the 15 topics above it belongs to, re-derive the relevant formula/rule, and work through the specific numbers given in that question step by step (as shown in the worked examples). Most "trick" questions test whether you know *which* method/attribute does what — not complex math.
