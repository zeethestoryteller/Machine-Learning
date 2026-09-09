Polynomial regression models non-linear relationships between features and labels by combining a polynomial feature transformation with a linear regression model. Because it uses more parameters due to the polynomial representation of inputs, it is more prone to overfitting.

**Core Implementation Steps**

* **Step 1:** Apply polynomial transformation on the feature matrix using `PolynomialFeatures`.


* **Step 2:** Learn a linear regression model (via normal equation or Stochastic Gradient Descent) on the transformed feature matrix.


* **Implementation Tip:** Make use of the `Pipeline` construct to chain polynomial transformation followed by a linear regression estimator.



**Key Parameters of `PolynomialFeatures**`

* **`degree`**: Controls the power/degree of the polynomial features (e.g., `degree=2`). It serves as a hyperparameter that can be tuned.


* **`interaction_only`**: A boolean parameter (default is `False`). When set to `True`, it excludes powers of single features (like $x_1^2$) and only produces interaction features (multiplying distinct features like $x_1 x_2$). For example, input $[x_1, x_2]$ transforms to $[1, x_1, x_2, x_1 x_2]$ instead of including individual squared terms.



**Code Implementation: Normal Equation**

```python
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

poly_model = Pipeline([
    ('polynomial_transform', PolynomialFeatures(degree=2)),
    ('linear_regression', LinearRegression())
])
poly_model.fit(X_train, y_train)

```

**Code Implementation: SGD Regressor**

```python
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

poly_model = Pipeline([
    ('polynomial_transform', PolynomialFeatures(degree=2)),
    ('sgd_regression', SGDRegressor())
])
poly_model.fit(X_train, y_train)

```

**Hyperparameter Tuning: Determining the Polynomial Degree via Grid Search**
You can use `GridSearchCV` combined with a pipeline to find the optimal polynomial degree:

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import SGDRegressor

param_grid = [
    {'poly_degree': [2, 3, 4, 5, 6, 7, 8, 9]}
]

pipeline = Pipeline(steps=[
    ('poly', PolynomialFeatures()),
    ('sgd', SGDRegressor())
])

grid_search = GridSearchCV(
    pipeline, 
    param_grid, 
    cv=5, 
    scoring='neg_mean_squared_error', 
    return_train_score=True
)
grid_search.fit(X_train.reshape(-1, 1), y_train)

```

**Adding Regularization to Polynomial Regression**
Because polynomial models easily overfit, regularization can be integrated into the pipeline:

* **Ridge Regularization:** Chain `PolynomialFeatures` with a `Ridge` estimator (using the `alpha` parameter for regularization rate) or `SGDRegressor` with `penalty='l2'`.


* **Lasso Regularization:** Chain `PolynomialFeatures` with a `Lasso` estimator (using `alpha`) or `SGDRegressor` with `penalty='l1'`.


* **ElasticNet Regularization:** Chain `PolynomialFeatures` with `SGDRegressor(penalty='elasticnet', l1_ratio=0.3)` to combine L1 and L2 penalties.
