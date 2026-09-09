# Multilayer Perceptron (MLP) — Complete Guide with sklearn

Based on: *Neural Networks*, Dr. Ashish Tendulkar, IIT Madras (Machine Learning Practice)

---

## 1. What is an MLP?

- MLP is a **supervised learning algorithm**.
- It learns a **non-linear function approximator** — it can be used for either **classification** or **regression**, depending on the dataset you give it.
- In `sklearn`, MLP is implemented via two classes (in `sklearn.neural_network`):

| Task | Class |
|---|---|
| Classification | `MLPClassifier` |
| Regression | `MLPRegressor` |

- `MLPClassifier` supports:
  - **Multi-class classification** — via a **Softmax** output layer.
  - **Multi-label classification** — a sample can belong to more than one class simultaneously.
- `MLPRegressor` supports:
  - **Multi-output regression** — a sample can have more than one target value.

### How an MLP is structured (conceptually)
An MLP is a stack of layers: **input layer → one or more hidden layers → output layer**. Each hidden layer applies a linear transformation (weights + bias) followed by a non-linear **activation function**. The network is trained by minimizing a loss function using **backpropagation** combined with an optimizer (solver).

---

## 2. Training Data Format

| Array | Holds | Shape |
|---|---|---|
| `X` | Training samples (features) | `(n_samples, n_features)` |
| `y` | Targets/labels | `(n_samples,)` (or `(n_samples, n_outputs)` for multi-label/multi-output) |

---

## 3. MLPClassifier — Step by Step

### Step 1: Instantiate the estimator
```python
from sklearn.neural_network import MLPClassifier
MLP_clf = MLPClassifier()
```

### Step 2: Fit the model
```python
# Model training with feature matrix X_train and label vector/matrix y_train
MLP_clf.fit(X_train, y_train)
```

### Step 3: Predict on new data
```python
MLP_clf.predict(X_test)         # returns class labels, e.g. array([1, 0])
MLP_clf.predict_proba(X_test)   # returns probability estimates, e.g. array([1.967e-04, 9.998e-01])
```
> **Note:** `MLPClassifier` supports only the **Cross-Entropy loss function**.

---

## 4. MLPRegressor — Step by Step

`MLPRegressor` trains using backpropagation, but with **no activation function on the output layer** (i.e., identity/linear output). Because of this:
- It uses **squared error** as the loss function.
- The output is a set of **continuous values**.
- **All the parameters of `MLPRegressor` are identical to those of `MLPClassifier`** (see full parameter table below).

### Step 1: Instantiate the estimator
```python
from sklearn.neural_network import MLPRegressor
MLP_reg = MLPRegressor()
```

### Step 2: Fit the model
```python
MLP_reg.fit(X_train, y_train)
```

### Step 3: Predict / Score
```python
MLP_reg.predict(X_test)          # returns continuous predictions, e.g. array([-0.9..., -7.1...])
MLP_reg.score(X_test, y_test)    # returns R² score, e.g. 0.45678889
```

---

## 5. Full Parameter Reference

### `hidden_layer_sizes`
Controls the network **architecture**: number of hidden layers and neurons per layer.
- It's a **tuple**: the *i*-th element = number of neurons in the *i*-th hidden layer.
- **Length of the tuple = number of hidden layers.**

```python
# 3 hidden layers: 15 neurons -> 10 neurons -> 5 neurons
MLPClassifier(hidden_layer_sizes=(15, 10, 5))
```

### `activation` (activation function for hidden layers)

| Value | Function | Formula |
|---|---|---|
| `'identity'` | no-op | f(x) = x |
| `'logistic'` | logistic sigmoid | f(x) = 1 / (1 + exp(-x)) |
| `'tanh'` | hyperbolic tangent | f(x) = tanh(x) |
| `'relu'` (**default**) | rectified linear unit | f(x) = max(0, x) |

```python
MLPClassifier(activation='relu')   # default
```

### `alpha` — Regularization
- Sets the **L2 penalty (regularization)** strength.
- A `float` value.
- **Default:** `alpha = 0.0001`

```python
MLPClassifier(alpha=0.0001)
```

### `solver` — Weight optimization algorithm
MLP optimizes the **log-loss function** using one of:

| Value | Description |
|---|---|
| `'lbfgs'` | An optimizer from the quasi-Newton family; **does not use minibatches** |
| `'sgd'` | Stochastic gradient descent |
| `'adam'` (**default**) | Stochastic gradient-based optimizer (works well on large datasets) |

```python
MLPClassifier(solver='adam')   # default
```

### `batch_size`
- Size of minibatches used by stochastic optimizers (`'sgd'` or `'adam'`). Irrelevant for `'lbfgs'`.
- **Default:** `'auto'` → `batch_size = min(200, n_samples)`

```python
MLPClassifier(batch_size='auto')
```

### Learning-rate-related parameters (used only with `solver='sgd'`, unless noted)

| Parameter | Meaning | Values | Default |
|---|---|---|---|
| `learning_rate` | Learning rate **schedule** for weight updates. Used only when `solver='sgd'`. | `'constant'`, `'invscaling'`, `'adaptive'` | `'constant'` |
| `learning_rate_init` | Initial learning rate. Used when `solver='sgd'` or `'adam'`. | float | `0.001` |
| `power_t` | Exponent for `invscaling` learning rate. Used only when `solver='sgd'`. | float | `0.5` |
| `max_iter` | Maximum number of training iterations (epochs). | int | `500` |
| `shuffle` | Whether to shuffle samples in each iteration. Used when `solver='sgd'` or `'adam'`. | bool | — |
| `momentum` | Momentum for gradient descent update. Used only when `solver='sgd'`. | float (0–1) | — |

```python
MLPClassifier(
    learning_rate='constant',
    learning_rate_init=0.001,
    power_t=0.5,
    max_iter=500,
    shuffle=True,
    momentum=0.9,
    solver='sgd'
)
```

### Full parameter summary table

| Parameter | Purpose | Default |
|---|---|---|
| `hidden_layer_sizes` | Network architecture (layers & neurons) | `(100,)` |
| `activation` | Non-linearity in hidden layers | `'relu'` |
| `solver` | Weight optimization algorithm | `'adam'` |
| `alpha` | L2 regularization strength | `0.0001` |
| `batch_size` | Minibatch size (sgd/adam only) | `'auto'` |
| `learning_rate` | LR schedule (sgd only) | `'constant'` |
| `learning_rate_init` | Initial LR (sgd/adam) | `0.001` |
| `power_t` | Exponent for invscaling LR (sgd only) | `0.5` |
| `max_iter` | Max training epochs | `500` |
| `shuffle` | Shuffle samples each epoch (sgd/adam) | `True` |
| `momentum` | Momentum term (sgd only) | `0.9` |

---

## 6. Inspecting a Trained Model

### `coefs_` — Weight matrices
- A **list** of shape `(n_layers - 1,)`.
- The *i*-th element = the weight matrix connecting layer *i* to layer *i+1*.

```python
print(MLP_clf.coefs_[0])   # weights between input layer and 1st hidden layer
print(MLP_clf.coefs_[1])   # weights between 1st hidden layer and 2nd hidden layer
```

### `intercepts_` — Bias vectors
- A **list** of shape `(n_layers - 1,)`.
- The *i*-th element = bias vector for layer *i+1*.

```python
print(MLP_clf.intercepts_[0])   # bias values for 1st hidden layer
print(MLP_clf.intercepts_[1])   # bias values for 2nd hidden layer
```

---

## 7. Complete Worked Example — Classification

```python
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 1. Load data
X, y = load_breast_cancer(return_X_y=True)

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. IMPORTANT: MLPs are sensitive to feature scale — always standardize!
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Instantiate the model with explicit hyperparameters
MLP_clf = MLPClassifier(
    hidden_layer_sizes=(64, 32),   # 2 hidden layers: 64 -> 32 neurons
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size='auto',
    learning_rate_init=0.001,
    max_iter=500,
    shuffle=True,
    random_state=42
)

# 5. Train
MLP_clf.fit(X_train, y_train)

# 6. Predict
y_pred = MLP_clf.predict(X_test)
y_proba = MLP_clf.predict_proba(X_test)

# 7. Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 8. Inspect learned parameters
print("Number of layers:", MLP_clf.n_layers_)
print("Weight matrix shapes:", [w.shape for w in MLP_clf.coefs_])
print("Bias vector shapes:", [b.shape for b in MLP_clf.intercepts_])
```

---

## 8. Complete Worked Example — Regression

```python
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load data
X, y = fetch_california_housing(return_X_y=True)

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Scale features (and often helps to scale targets too, for regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Instantiate
MLP_reg = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    alpha=0.001,
    max_iter=1000,
    random_state=42
)

# 5. Train
MLP_reg.fit(X_train, y_train)

# 6. Predict & score
y_pred = MLP_reg.predict(X_test)
print("R^2 score:", MLP_reg.score(X_test, y_test))
print("R^2 (manual):", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
```

---

## 9. Practical Tips for Understanding & Using MLPs

1. **Always scale your features.** MLPs (unlike trees) are very sensitive to feature magnitude — use `StandardScaler` or `MinMaxScaler` before fitting.
2. **`hidden_layer_sizes` is your architecture dial.** Start small (e.g., one layer of 50–100 neurons) and increase complexity only if underfitting.
3. **`solver='adam'`** is the best default for most datasets. Use `'lbfgs'` for small datasets (it can converge faster and more reliably on small data). Use `'sgd'` if you want fine control over the learning-rate schedule and momentum.
4. **`alpha` controls overfitting.** If your model overfits (great train score, poor test score), increase `alpha`. If it underfits, decrease it.
5. **`max_iter`**: if you get a `ConvergenceWarning`, increase `max_iter`.
6. **`coefs_` and `intercepts_`** let you literally see the learned weights/biases — useful for debugging or explaining what the network learned.
7. **Classification vs regression** — the only structural difference: `MLPClassifier` ends with a Softmax (or sigmoid for binary/multi-label) and uses cross-entropy loss; `MLPRegressor` ends with an identity (linear) output and uses squared-error loss. Every other parameter behaves identically between the two classes.

---

## 10. Quick Recap Table

| Concept | Classifier | Regressor |
|---|---|---|
| Class | `MLPClassifier` | `MLPRegressor` |
| Output activation | Softmax (multi-class) / Sigmoid | Identity (linear) |
| Loss function | Cross-Entropy | Squared Error |
| Output | Class labels / probabilities | Continuous values |
| Multi-target support | Multi-label classification | Multi-output regression |
| `.fit(X, y)` | ✅ | ✅ |
| `.predict(X)` | class labels | continuous values |
| `.predict_proba(X)` | ✅ | ❌ (not applicable) |
| `.score(X, y)` | accuracy | R² |
| Parameters | Same set | Same set |
