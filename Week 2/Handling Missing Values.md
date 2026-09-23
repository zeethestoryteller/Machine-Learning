# Guide to Handling Missing Values 

## Why missing values occur
Missing values typically come from errors in data capture — sensor malfunctioning, measurement errors, etc. Many ML algorithms can't handle missing data directly, but you also shouldn't just discard those records since that wastes valuable training samples. `sklearn.impute` is the module built for filling these gaps.

Three main tools:
- **SimpleImputer** — fills with a simple statistic
- **KNNImputer** — fills using nearest neighbors
- **MissingIndicator** — flags where values were missing

---

## 1. SimpleImputer

Fills missing values using one of four strategies: `'mean'`, `'median'`, `'most_frequent'`, or `'constant'`.

**Example:** Given
```
X = [[7,   1],
     [nan, 8],
     [2,   nan],
     [9,   6]]
```
Using `strategy='mean'`: column 1 mean = (7+2+9)/3 = 6, column 2 mean = (1+8+6)/3 = 5

```python
from sklearn.impute import SimpleImputer

si = SimpleImputer(strategy='mean')
X_transformed = si.fit_transform(X)
```
Result:
```
[[7, 1],
 [6, 8],
 [2, 5],
 [9, 6]]
```

---

## 2. KNNImputer

Fills a missing value using the **mean of the same attribute from its n_neighbors nearest neighbors** (by Euclidean distance).

**Distance with missing coordinates** is computed using a weighted formula:

$$\text{dist}(x,y) = \sqrt{\text{weight} \times \text{distance from present coordinates}^2}$$

where:
$$\text{weight} = \frac{\text{total number of coordinates}}{\text{number of present coordinates}}$$

Example: distance between `x=[3, nan, nan, 6]` and `y=[1, nan, 4, 5]`:
- weight = 4/2 = 2
- distance² from present coords = (3−1)² + (6−5)² = 5
- dist = √(2×5) ≈ 3.3

**Worked example:** Given
```
X = [[1.,   2.,  nan],
     [3.,   4.,  3.],
     [nan,  6.,  5.],
     [8.,   8.,  7.]]
```
To fill row 1's missing value, distances from `[1, 2, nan]` to the other rows are computed (≈3.46, ≈6.92, ≈11.29). The 2 nearest neighbors are rows `[3,4,3]` and `[nan,6,5]`, whose 3rd-column values are 3 and 5 → mean = 4.

```python
from sklearn.impute import KNNImputer

knni = KNNImputer(n_neighbors=2, weights="uniform")
X_transformed = knni.fit_transform(X)
```
Result:
```
[[1.,  2.,  4. ],
 [3.,  4.,  3. ],
 [5.5, 6.,  5. ],
 [8.,  8.,  7. ]]
```

---

## 3. MissingIndicator

Sometimes it's useful to **track where values were missing**, even after imputing them. `MissingIndicator` returns a binary matrix where `True` marks an originally-missing entry.

```python
from sklearn.impute import MissingIndicator

mi = MissingIndicator()
missing_mask = mi.fit_transform(X)
```

---

## Practical tip: combine imputers with pipelines

Since the **same preprocessing must be applied identically to train/eval/test sets**, it's best practice to wrap imputers inside a `Pipeline`:

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

X_transformed = pipe.fit_transform(X_train)
X_test_transformed = pipe.transform(X_test)  # same transformation applied
```

This avoids code duplication and prevents distribution shift/inconsistent preprocessing between train and test sets.

