# SatadaredScaler:

$$z = \frac{X_{\text{new}} - \mu}{\sigma}$$

# MinMaxScaler:

Xnew = (X- Min) / (Max - Min)

# MaxAbsScaler:

Xnew = X / max absolute value of the column


# RoubustScaler:

Xnew = (X-Q2) / (Q3- Q1)

# Compute the Recall Score:

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} $$

# Cosine Similarity formula:

$$\text{Cosine Similarity} = \frac{A \cdot B}{\Vert{}A\Vert{} \Vert{}B\Vert{}}$$


# DummyRegressor:

Ypred = mean, Median value of the Y_label

---
### Linear Regressor:

Y_pred = W0 + W1X1 + W2X2 + ... = XW

$$ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2 $$

Jahan:
$n$: Total number of data points (samples)

$Y_i$: Actual value (true label)

$\hat{Y}_i$: Predicted value (Y_pred)

---
### Gradient Dessent:

**Gradient Descent** ka weight update formula:

$$W = W - \alpha \frac{\partial J}{\partial W}$$

* Derivative of Cost Function (MSE)

**Summation Form:**

$$ \frac{\partial J}{\partial W} = \frac{2}{n} \sum_{i=1}^{n} X_i (\hat{Y}_i - Y_i) $$

**Matrix Form:**

$$ \frac{\partial J}{\partial W} = \frac{2}{n} X^T (\hat{Y} - Y) $$

Jahan:

* **$W$**: Current weight (parameter)
* **$\alpha$** (Alpha): Learning rate (jo step size decide karta hai)
* **$J$**: Cost function (jaise MSE ya Loss)
* **$\frac{\partial J}{\partial W}$**: Cost function ka gradient ya derivative weights ke respect mein
---
### Regularization

**L1 Regularization (Lasso):**

$$ J_{L1} = J + \lambda \sum_{j=1}^{p} |W_j| $$

**L2 Regularization (Ridge):**

$$ J_{L2} = J + \lambda \sum_{j=1}^{p} W_j^2 $$

Jahan:
* **$\lambda$** (Lambda): Regularization parameter
* **$W_j$**: Weights
---
#### Logistic Regression

**Hypothesis (Sigmoid Function):**

$$ \hat{Y} = \frac{1}{1 + e^{-W^T X}} $$

**Cost Function (Binary Cross-Entropy / Log Loss):**

$$ J = -\frac{1}{n} \sum_{i=1}^{n} \left[ Y_i \log(\hat{Y}_i) + (1 - Y_i) \log(1 - \hat{Y}_i) \right] $$

Jahan:
* **$\hat{Y}$**: Predicted probability (0 to 1)
* **$W$**: Weights 
* **$X$**: Input features
* **$Y_i$**: Actual true label (0 or 1)
* **$n$**: Total number of samples
---
### K-Nearest Neighbors (KNN) Distance Metrics

**1. Euclidean Distance (Most common, L2 Norm):**

$$ d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2} $$

**2. Manhattan Distance (L1 Norm):**

$$ d(x, y) = \sum_{i=1}^{n} |x_i - y_i| $$

**3. Minkowski Distance (Generalized form):**

$$ d(x, y) = \left( \sum_{i=1}^{n} |x_i - y_i|^p \right)^{\frac{1}{p}} $$

Jahan:
* **$x, y$**: Two data points (vectors) jinke beech ka distance calculate karna hai
* **$n$**: Total number of features (dimensions)
* **$p$**: Minkowski parameter ($p=1$ ho toh Manhattan, $p=2$ ho toh Euclidean)
* **$K$**: Number of nearest neighbors (hyperparameter jo hum set karte hain)
---
### Naive Bayes Classifier

**Bayes' Theorem:**

$$ P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)} $$

**Naive Bayes Equation (Multiple Features):**

$$ P(y | x_1, x_2, \dots, x_n) \propto P(y) \prod_{i=1}^{n} P(x_i | y) $$

**Final Prediction (Class with maximum probability):**

$$ \hat{y} = \arg\max_y P(y) \prod_{i=1}^{n} P(x_i | y) $$

Jahan:
* **$P(y | x_1, \dots, x_n)$**: Posterior probability (target class $y$ ki probability given the features)
* **$P(y)$**: Prior probability (class $y$ ki overall probability)
* **$P(x_i | y)$**: Likelihood (feature $x_i$ ki probability given class $y$)
* **$\hat{y}$**: Predicted class jo maximum probability rakhti hai
---
# Support Vector Machine (SVM)

**1. Decision Boundary (Hyperplane):**

$$ w^T x + b = 0 $$

**2. Cost Function (Hinge Loss with L2 Regularization):**

$$ J(w, b) = \frac{1}{2} ||w||^2 + C \sum_{i=1}^{n} \max(0, 1 - y_i (w^T x_i + b)) $$

**3. Prediction Rule:**

$$ \hat{y} = \text{sign}(w^T x + b) $$

*(Agar value > 0 hai toh Class +1, warna Class -1)*

Jahan:
* **$w$**: Weight vector (margin ke perpendicular direction)
* **$b$**: Bias (hyperplane ka offset)
* **$x_i$**: Input features (data points)
* **$y_i$**: Actual true label (SVM mein aam taur par -1 ya 1 hota hai)
* **$C$**: Regularization parameter (Margin size aur misclassification ke beech ka trade-off)
* **$\frac{1}{2} ||w||^2$**: Margin maximization term (L2 Regularization)
* **$\max(0, 1 - y_i (w^T x_i + b))$**: Hinge Loss (Galat classification par penalty)

