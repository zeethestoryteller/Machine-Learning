**1. Ridge Classifier** (for Least Square Classification)

* **Overview:** It is a classifier variant of the Ridge regressor (`RidgeClassifier` from `sklearn.linear_model`). It first converts binary targets to $\{-1, 1\}$ and then treats the classification task as a regression problem.
* **Objective Function:** Minimizes a penalized residual sum of squares:

$$\min_{w} \vert{}\vert{}Xw - y\vert{}\vert{}_2^2 + \alpha\vert{}\vert{}w\vert{}\vert{}_2^2$$



where $\alpha$ denotes the regularization rate. The predicted class corresponds to the sign of the regressor's prediction.
* **Key Implementation Steps:**
1. **Instantiation:**
```python
from sklearn.linear_model import RidgeClassifier
ridge_classifier = RidgeClassifier()

```


2. **Training:**
```python
ridge_classifier.fit(X_train, y_train)

```

3. **Predictions**

```python
# Predict labels for feature matrix X_test
y_pred = ridge_classifier.predict(X_test)


```



### Customization & Parameters
* **Regularization ($\alpha$):** Set via `alpha` (default is `0.1`). Must be positive; larger values specify stronger regularization.
```python
ridge_classifier = RidgeClassifier(alpha=0.001)
```
* **Solvers:** Optimization can be configured using the `solver` parameter.
```python
ridge_classifier = RidgeClassifier(solver=auto)
```
  * `'auto'`:  By default, it uses `'auto'`.
  * `'svd'`: uses a Singular Value Decomposition of the feature matrix to
compute the Ridge coefficients.

  * `'cholesky'`,
  * `'sparse_cg'`: For Large-Scale Data
  * `'lsqr'`: uses the dedicated regularized least-squares routine
`scipy.sparse.linalg.lsqr`  and it is fastest.
  * `'sag', 'saga' `: For Large Datasets, When both $n_{samples}$ and $n_{features}$ are large
  * `'lbfgs'`:For Small Datasets
* **Intercept:** Controlled via `fit_intercept` (default is `True`). Set to `False` if the data is already centered.
```python
ridge_classifier = RidgeClassifier(fit_intercept=True)
```
