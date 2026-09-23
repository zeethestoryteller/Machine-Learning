# Guide to Categorical Transformers / Encoding (from the lecture)

Categorical (non-numeric) attributes need to be converted into sensible numerical representations before most ML algorithms can use them. Sklearn provides several encoders, each suited to a different situation — features vs. labels, single-label vs. multi-label, ordinal vs. nominal.

---

## 1. OneHotEncoder

Encodes a categorical **feature or label** as a one-hot numeric array — creates one binary column for each of *K* unique values, with exactly one column set to 1 per row.

**Example:**
```
x = [1, 2, 3, 1]
```
K = 3 unique values → 3 columns

```python
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()
X_transformed = ohe.fit_transform(x)
```
Result:
```
[[1, 0, 0],
 [0, 1, 0],
 [0, 0, 1],
 [1, 0, 0]]
```

**Use when:** encoding nominal categorical *features* (no inherent order) for most ML models.

---

## 2. LabelEncoder

Encodes **target labels** (y, 1D only) with values between 0 and K−1, where K is the number of distinct values.

**Example:**
```
y = [1, 2, 6, 1, 8, 6]
```
K = 4 distinct values: {1, 2, 6, 8} → encoded as 0, 1, 2, 3 respectively

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_transformed = le.fit_transform(y)
```
Result: `[0, 1, 2, 0, 3, 2]`

**Use when:** encoding a single-column target/label vector — **not** for feature matrices.

---

## 3. OrdinalEncoder

Similar to LabelEncoder (values between 0 and K−1), but works on **multi-dimensional feature data**, unlike LabelEncoder which is 1D-only.

**Example:**
```
X = [[1, 'male'],
     [2, 'female'],
     [6, 'female'],
     [1, 'male'],
     [8, 'male'],
     [6, 'female']]
```

```python
from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder()
X_transformed = oe.fit_transform(X)
```
Result:
```
[[0, 1],
 [1, 0],
 [2, 0],
 [0, 1],
 [3, 1],
 [2, 0]]
```

**Use when:** encoding categorical *feature columns* (as opposed to a single target), especially if there's a meaningful order to preserve (e.g., 'low', 'medium', 'high').

---

## 4. LabelBinarizer

Used when extending binary classifiers/regressors to multi-class in a **one-vs-all** fashion — converts multi-class labels into binary label matrix form.

**Example:**
```
y = [1, 2, 6, 1, 8, 6]
```

```python
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()
Y_transformed = lb.fit_transform(y)
```
Result (4 classes → 4 columns):
```
[[1, 0, 0, 0],
 [0, 1, 0, 0],
 [0, 0, 1, 0],
 [1, 0, 0, 0],
 [0, 0, 0, 1],
 [0, 0, 1, 0]]
```

**Note:** If your estimator already supports multiclass data natively, LabelBinarizer isn't needed.

---

## 5. MultiLabelBinarizer

Encodes **multi-label** categorical data (where each sample can belong to multiple categories at once) into a 0/K−1 binary matrix.

**Example:**
```python
movie_genres = [
    {'action', 'comedy'},
    {'comedy'},
    {'action', 'thriller'},
    {'science-fiction', 'action', 'thriller'}
]

from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
X_transformed = mlb.fit_transform(movie_genres)
```
Result (K=4 genres → 4 columns):
```
[[1, 1, 0, 0],
 [0, 1, 0, 0],
 [1, 0, 0, 1],
 [1, 0, 1, 1]]
```

**Use when:** each sample can have multiple tags/labels simultaneously (e.g., movie genres, multi-topic tags).

---

## Quick decision guide

| Situation | Encoder |
|---|---|
| Nominal categorical **feature**, want dummy columns | OneHotEncoder |
| Single-column **target/label**, need 0..K−1 | LabelEncoder |
| Multi-column categorical **features**, need 0..K−1 | OrdinalEncoder |
| Multi-class target → one-vs-all binary columns | LabelBinarizer |
| Each sample has **multiple** category memberships | MultiLabelBinarizer |

---

## Applying different encoders to different columns: ColumnTransformer

Real datasets usually mix numeric and categorical columns. Use `ColumnTransformer` (from `sklearn.compose`) to apply the right transformer to the right column(s):

**Example:**
```
X = [[20.0, 'male'],
     [11.2, 'female'],
     [15.6, 'female'],
     [13.0, 'male'],
     [18.6, 'male'],
     [16.4, 'female']]
```

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder

column_trans = ColumnTransformer(
    [('ageScaler', MaxAbsScaler(), [0]),
     ('genderEncoder', OneHotEncoder(dtype='int'), [1])],
    remainder='drop',
    verbose_feature_names_out=False
)
X_transformed = column_trans.fit_transform(X)
```
Result:
```
[[1.,   0., 1.],
 [0.56, 1., 0.],
 [0.78, 1., 0.],
 [0.65, 0., 1.],
 [0.93, 0., 1.],
 [0.82, 1., 0.]]
```
- Each tuple format: `('name', transformer(...), column_indices)`
- `remainder='drop'` drops any columns not explicitly listed (use `'passthrough'` to keep them unchanged).

---

## Best practice: wrap in a pipeline

As always, fit the encoder only on training data, then reuse it on test data:

```python
from sklearn.pipeline import Pipeline

pipe = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

X_train_encoded = pipe.fit_transform(X_train)
X_test_encoded = pipe.transform(X_test)
```

---

Want a follow-up guide on **feature selection** (VarianceThreshold, SelectKBest, RFE, etc.) or **pipelines & ColumnTransformer** in more depth next?
