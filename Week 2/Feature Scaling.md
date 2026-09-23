# Guide to Feature Scaling 

## Why scale features?
Numerical features with different scales cause **slower convergence** in iterative optimization procedures (like gradient descent). It's good practice to bring all numerical features onto the same scale before training. Sklearn offers three main scaling APIs, all found in `sklearn.preprocessing`.

---

## 1. StandardScaler

Transforms features to have **mean = 0** and **standard deviation = 1** (standardization / z-score normalization):

$$x' = \frac{x - \mu}{\sigma}$$

**Example:**
```
x = [4, 3, 2, 5, 6]
```
Here μ = 4, σ = √2

```python
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
x_transformed = ss.fit_transform(x)
```
Result: `[0, -1/√2, -2/√2, 1/√2, 2/√2]` — mean becomes 0, std becomes 1.

**Use when:** your data is roughly Gaussian, or you're using models sensitive to feature variance (e.g., linear models, SVMs, PCA).

---

## 2. MinMaxScaler

Transforms features so all values fall within **[0, 1]**:

$$x' = \frac{x - x.min}{x.max - x.min}$$

**Example:**
```
x = [15, 2, 5, -2, -5]
```
x.max = 15, x.min = -5

```python
from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
x_transformed = mms.fit_transform(x)
```
Result: `[1, 0.35, 0.5, 0.6, 0]` — the largest value maps to 1, the smallest to 0.

**Use when:** you want bounded values (e.g., for neural networks or algorithms that expect a fixed range), and you don't have extreme outliers.

---

## 3. MaxAbsScaler

Transforms features so all values fall within **[-1, 1]**, scaling by the maximum absolute value:

$$x' = \frac{x}{\text{MaxAbsoluteValue}}, \quad \text{MaxAbsoluteValue} = \max(x.max, |x.min|)$$

**Example:**
```
x = [4, 2, 5, -2, -100]
```
MaxAbsoluteValue = max(5, |-100|) = 100

```python
from sklearn.preprocessing import MaxAbsScaler

mas = MaxAbsScaler()
x_transformed = mas.fit_transform(x)
```
Result: `[0.04, 0.02, 0.05, -0.02, -1]`

**Use when:** your data is already centered at zero or is sparse (it preserves sparsity since it doesn't shift/center the data — no subtraction involved).

---

## Quick comparison

| Scaler | Output range | Sensitive to outliers? | Preserves sparsity? |
|---|---|---|---|
| StandardScaler | unbounded (mean 0, std 1) | Yes | No (centers data) |
| MinMaxScaler | [0, 1] | Very (min/max driven) | No |
| MaxAbsScaler | [-1, 1] | Yes | Yes |

---

## Related numeric transformer: FunctionTransformer

Not a "scaler" per se, but useful for custom transformations like log-scaling:

```python
import numpy
from sklearn.preprocessing import FunctionTransformer

ft = FunctionTransformer(numpy.log2)
X_transformed = ft.fit_transform(X)
```
This applies `log2` to every value in the feature matrix — useful when features span several orders of magnitude.

---

## Best practice: use pipelines

As with imputation, the **same scaling parameters (μ, σ, min, max) learned on the training set** must be applied to validation/test data — never refit the scaler on test data:

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

X_train_scaled = pipe.fit_transform(X_train)   # learns params from train
X_test_scaled = pipe.transform(X_test)         # reuses train's params
```

---

Want me to continue with a similar guide for **categorical encoding** (OneHotEncoder, OrdinalEncoder, LabelEncoder, etc.) or **feature selection** next?
