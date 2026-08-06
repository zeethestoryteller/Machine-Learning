**2. Perceptron**

* **Overview:** A simple linear classification algorithm suitable for large-scale learning. It shares the same underlying implementation as `SGDClassifier` and uses Stochastic Gradient Descent (SGD) for training.
* **Key Implementation Steps:**
1. **Instantiation:**
```python
from sklearn.linear_model import Perceptron
perceptron_classifier = Perceptron()

```


2. **Training:**
```python
perceptron_classifier.fit(X_train, y_train)

```

Perceptron classifier can be trained in an iterative manner with partial_fit method

* **Key Customization Parameters:**
* `penalty`: Regularization term (default is `'l2'`).
* `alpha`: Constant that multiplies the regularization term (default is `0.0001`).
* `eta0`: Constant learning rate (default is `1`).
* `max_iter`: Maximum number of passes over the training data (default is `1000`).
* `warm_start`: When set to `True`, reuses the solution of the previous call to fit as initialization (useful for iterative training with `partial_fit`).
* `fit_intercept`: default = True
* `n_iter_no_change`: default = 5
* `validation_fraction`: default = 0.1
* `early_stopping`: default = False
* `l1_ratio`: default = 0.15
* `tol`: default = 1e-3
