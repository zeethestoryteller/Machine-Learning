# Guide to Pipelines, FeatureUnion, ColumnTransformer & Chaining Transformers

Real-world ML workflows require applying multiple transformations in sequence, applying different transformations to different columns, and combining outputs from multiple transformers — all while guaranteeing the *exact same* preprocessing is applied consistently to train, validation, and test sets. Sklearn's `sklearn.pipeline` and `sklearn.compose` modules solve this.

---

## Why chaining matters

Manually chaining transformers looks like this:
```python
si = SimpleImputer()
X_imputed = si.fit_transform(X)

ss = StandardScaler()
X_scaled = ss.fit_transform(X_imputed)
```

**The critical rule:** the exact same transformation, in the exact same order, must be applied to training, evaluation, and test sets. Failing to do this causes **distribution shift** → incorrect predictions → incorrect performance evaluation. This is exactly what `Pipeline` automates and enforces.

---

## 1. Pipeline (`sklearn.pipeline.Pipeline`)

Sequentially applies a list of transformers, ending in an optional final estimator.

- **Intermediate steps** must be transformers (implement `fit` and `transform`)
- **Final step** only needs to implement `fit` (can be a transformer or a predictor like a regressor/classifier)
- Purpose: bundle multiple steps so they can be **cross-validated together** while tuning parameters jointly

### Creating a Pipeline — two ways

**Method 1: `Pipeline()` with named steps**
```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

estimators = [
    ('simpleImputer', SimpleImputer()),
    ('standardScaler', StandardScaler()),
]
pipe = Pipeline(steps=estimators)
pipe.fit_transform(X)
```

**Method 2: `make_pipeline()` — auto-names steps**
```python
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(SimpleImputer(), StandardScaler())
```

Both are equivalent to the manual chaining shown above — but now the whole chain is a single object exposing the interface of its last step.

#

### Accessing individual steps

```python
estimators = [
    ('simpleImputer', SimpleImputer()),
    ('pca', PCA()),
    ('regressor', LinearRegression())
]
pipe = Pipeline(steps=estimators)
```

The `pca` step can be accessed 4 equivalent ways:
```python
pipe.named_steps.pca
pipe.steps[1]
pipe[1]
pipe['pca']
```

#

### Setting parameters of individual steps

Use the `<estimatorName>__<parameterName>` syntax (note: **two underscores**):

```python
pipe.set_params(pca__n_components=2)
```
#
### Advantages of Pipeline
- Combines multiple steps (imputation, scaling, encoding, model training, CV) into a **single object**
- Enables **joint grid search** across all steps' parameters
- Offers convenience — call `fit()`/`predict()` only on the `Pipeline` object
- **Reduces code duplication** — no need to repeat preprocessing code for test data


#
### Caching transformers (avoid redundant computation)

Transforming data can be expensive. During grid search, you often don't need to *re-run* transformers for every parameter combination — the `memory` parameter caches transformer outputs:

```python
estimators = [
    ('simpleImputer', SimpleImputer()),
    ('pca', PCA(2)),
    ('regressor', LinearRegression())
]
pipe = Pipeline(steps=estimators, memory='/path/to/cache/dir')
```
`memory` accepts a directory path (string) or a `joblib.Memory` object.

---

## 2. ColumnTransformer (`sklearn.compose.ColumnTransformer`)

Applies **different transformers to different columns/subsets of features**, then concatenates the results into a single matrix. Essential for heterogeneous data (mix of numeric + categorical columns).

**Format:** each tuple is `('estimatorName', estimator(...), columnIndices)`

**Example:** weight (numeric) + gender (categorical) columns
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

- `remainder='drop'` (default) discards any columns not explicitly listed; use `'passthrough'` to keep them unchanged.
- Combines different feature selection/transformation mechanisms into a single transformer object.

---

## 3. TransformedTargetRegressor (`sklearn.compose`)

Transforms the **target variable `y`** before fitting a regression model, then automatically maps predictions back via an inverse transform.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor

tt = TransformedTargetRegressor(
    regressor=LinearRegression(),
    func=np.log,
    inverse_func=np.exp
)

X = np.arange(4).reshape(-1, 1)
y = np.exp(2 * X).ravel()
tt.fit(X, y)
```
**Use when:** your target variable benefits from a transformation (e.g., log-transforming a skewed target) but you still want predictions in the original scale.

---

## 4. FeatureUnion (`sklearn.pipeline.FeatureUnion`)

**Concatenates results of multiple transformer objects** — applies a list of transformers **in parallel** (not sequentially like Pipeline), and their outputs are joined side-by-side into a larger matrix.

```python
from sklearn.pipeline import FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelBinarizer

num_pipeline = Pipeline([
    ('selector', ColumnTransformer([
        ('select_first_4', 'passthrough', slice(0, 4))
    ])),
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

cat_pipeline = ColumnTransformer([
    ('label_binarizer', LabelBinarizer(), [4]),
])

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])
```

**Pipeline vs FeatureUnion:**
| | Pipeline | FeatureUnion |
|---|---|---|
| Execution | Sequential (step-by-step) | Parallel (side-by-side) |
| Output | Result of the last step | Concatenation of all outputs |
| Use case | Fixed sequence: preprocess → model | Combine multiple independent feature sets |

`FeatureUnion` and `Pipeline` can be **nested together** to build arbitrarily complex composite transformers, as shown above.

---

## 5. Visualizing composite transformers

Sklearn can render an interactive diagram of your pipeline structure in Jupyter:

```python
from sklearn import set_config
set_config(display='diagram')
full_pipeline  # displays HTML diagram in a Jupyter context
```

This produces a visual tree showing `FeatureUnion` → `num_pipeline` (selector → imputer → scaler) and `cat_pipeline` (label_binarizer) side by side — very useful for debugging complex nested pipelines.

---

## Summary: which tool to use when

| Goal | Tool |
|---|---|
| Apply transformers **in sequence** (preprocess → model) | `Pipeline` |
| Apply **different transformers to different columns** | `ColumnTransformer` |
| Transform the **target variable** for regression | `TransformedTargetRegressor` |
| Apply multiple transformers **in parallel** and concatenate outputs | `FeatureUnion` |
| Combine all of the above into one complex end-to-end object | Nest `Pipeline` + `ColumnTransformer` + `FeatureUnion` together |

---

## Putting it all together — full example

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression

numeric_features = [0, 1]
categorical_features = [2]

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

full_pipeline.fit(X_train, y_train)
predictions = full_pipeline.predict(X_test)
```

This single `full_pipeline` object handles missing-value imputation, scaling, encoding, and model fitting/prediction end-to-end — with guaranteed consistency between train and test preprocessing.
