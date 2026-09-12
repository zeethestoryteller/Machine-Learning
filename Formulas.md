# SatadaredScaler:

$$z = \frac{X_{\text{new}} - \mu}{\sigma}$$

# MinMaxScaler:

Xnew = (X- Min) / (Max - Min)

# MaxAbsScaler:

Xnew = X / max absolute value of the column


# RoubustScaler:

Xnew = (X-Q2) / (Q3- Q1)

---
# OneHotScaler:
<img width="1006" height="303" alt="image" src="https://github.com/user-attachments/assets/d12985ac-fa4b-4b70-bb4c-d80180b44f4b" />

---
# LabelEncoder:
<img width="1026" height="418" alt="image" src="https://github.com/user-attachments/assets/94a660ec-7efa-4ed9-9aeb-d4786bdb818f" />

---
# OrdinalEncoder:
<img width="995" height="261" alt="image" src="https://github.com/user-attachments/assets/d0c7d2c8-e60f-4b26-9248-56a0ac9cdc6d" />

---
# Label Binarizer:
<img width="1017" height="271" alt="image" src="https://github.com/user-attachments/assets/0e15cbff-135a-4650-93fe-1e5e2ceae101" />


---
# MultiLabelBinarizer

<img width="1012" height="487" alt="image" src="https://github.com/user-attachments/assets/85f9f7ac-a183-47f0-9b90-8d6d90400b7f" />

---
# Add Dummy Feature:
<img width="992" height="185" alt="image" src="https://github.com/user-attachments/assets/e65d0b91-ffed-48ad-bff7-c31ff4dd754d" />

---
# Recall Score:

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} $$

# Precision Score:
$$ \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} $$

# F1 Score:
$$ \text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} $$

Jahan:
* **TP**: True Positives (Correctly predicted positive observations)
* **FP**: False Positives (Incorrectly predicted positive observations)

# Cosine Similarity formula:

$$\text{Cosine Similarity} = \frac{A \cdot B}{\Vert{}A\Vert{} \Vert{}B\Vert{}}$$

---
**Term Frequency (TF)**
$TF(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}$

* $t$: The specific term or word.
* $d$: The specific document being evaluated.
* $f_{t,d}$: The raw count of times term $t$ appears in document $d$.
* The denominator represents the total number of words in document $d$.
* Measures how frequently a term occurs within a single document.

**Inverse Document Frequency (IDF)**
$IDF(t, D) = \log\left(\frac{N}{df_t}\right)$

* $N$: Total number of documents in the entire corpus $D$.
* $df_t$: The document frequency, or the number of documents in the corpus that contain the term $t$.
* Measures how much information the word provides. It penalizes highly frequent, generic words (like "the" or "and") across the corpus to highlight contextually significant terms. *(Note: Machine learning libraries like scikit-learn often apply smoothing by adding 1 to the numerator and denominator to prevent division by zero).*

**TF-IDF Score**
$TF\text{-}IDF(t, d, D) = TF(t, d) \times IDF(t, D)$

* The final weight assigned to term $t$ in document $d$.
* Yields a high score for a term that appears frequently in a specific document but rarely across the overall corpus, making it an excellent identifier for that document's unique subject matter.
---
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

# SVC (Support Vector Classification) - Dual Formulation**
SVC mein Kernel trick apply karne ke liye aam taur par Dual formulation ka use hota hai:

**Dual Objective Function:**


$$\max_{\alpha} \sum_{i=1}^{n} \alpha_i - \frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_i \alpha_j y_i y_j K(x_i, x_j)$$

**Decision Function (Prediction):**


$$\hat{y} = \text{sign} \left( \sum_{i=1}^{n} \alpha_i y_i K(x_i, x) + b \right)$$

Jahan:

* **$\alpha_i$**: Lagrange multipliers
* **$K(x_i, x_j)$**: Kernel function (jaise Linear, RBF, Polynomial) jo data ko higher dimension mein map karta hai
* **$x$**: Naya input data point
* **$y_i$**: True class labels (-1 ya 1)



# SVR (Support Vector Regression)**
SVR regression problems ke liye $\epsilon$-insensitive loss function ka use karta hai, jahan ek specific margin ($\epsilon$) ke andar aane wale errors ko ignore kiya jata hai.

**Cost Function (Primal Form):**


$$J(w, b) = \frac{1}{2} \vert{}\vert{}w\vert{}\vert{}^2 + C \sum_{i=1}^{n} \max(0, \vert{}y_i - (w^T x_i + b)\vert{} - \epsilon)$$

**Constraints (Slack Variables $\xi$ aur $\xi^*$ ke sath):**


$$\vert{}y_i - (w^T x_i + b)\vert{} \le \epsilon + \xi_i$$

Jahan:

* **$w$**: Weight vector
* **$b$**: Bias
* **$\epsilon$** (Epsilon): Margin of tolerance (is tube ke andar error par koi penalty nahi lagti)
* **$C$**: Regularization parameter (margin aur error tolerance ke beech ka trade-off)
* **$\max(0, \vert{}y_i - \hat{y}_i\vert{} - \epsilon)$**: $\epsilon$-insensitive loss function
* **$\xi_i$** (Xi): Slack variables (jo data points $\epsilon$-tube ke bahar hain unka error measure karne ke liye)

---

# Ridge Classifier

**Cost Function (L2 Regularized Least Squares):**

$$ J(W, b) = \sum_{i=1}^{n} (y_i - (W^T x_i + b))^2 + \alpha \sum_{j=1}^{p} W_j^2 $$

**Matrix Form:**

$$ J(W) = ||XW - Y||^2_2 + \alpha ||W||^2_2 $$

**Prediction Rule:**

$$ \hat{y} = \text{sign}(W^T x + b) $$

*(Agar output > 0 hai toh Class +1, warna Class -1)*

Jahan:
* **$W$**: Weights (parameters)
* **$b$**: Bias
* **$x_i$** / **$X$**: Input features
* **$y_i$** / **$Y$**: Target labels (converted to -1 aur 1)
* **$\alpha$** (Alpha): Regularization strength (penalty term jo overfitting rokti hai)
* **$p$**: Total number of features
* **$n$**: Total number of data points

---

# Perceptron

**1. Linear Output (Weighted Sum):**

$$ z = W^T X + b = \sum_{i=1}^{n} w_i x_i + b $$

**2. Activation Function (Heaviside Step Function):**

$$ \hat{y} = \begin{cases} 1 & \text{if } z \ge 0 \\ 0 & \text{if } z < 0 \end{cases} $$
*(Kahin-kahin labels -1 aur 1 bhi use hote hain, tab $\hat{y} = \text{sign}(z)$ hota hai)*

**3. Weight Update Rule:**

$$ W = W + \alpha (Y - \hat{Y}) X $$
$$ b = b + \alpha (Y - \hat{Y}) $$

Jahan:
* **$W$**: Weights vector
* **$b$**: Bias
* **$X$**: Input features
* **$z$**: Net input (weighted sum)
* **$\hat{Y}$**: Predicted output (0 ya 1)
* **$Y$**: Actual true label (0 ya 1)
* **$\alpha$** (Alpha): Learning rate

---
# Multi-Layer Perceptron (MLP)

**1. Linear Transformation (For Layer $l$):**

$$ z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)} $$

**2. Activation (For Layer $l$):**

$$ a^{(l)} = f(z^{(l)}) $$

**3. Final Prediction (Output Layer $L$):**

$$ \hat{y} = a^{(L)} $$

Jahan:
* **$l$**: Current layer index (1 se lekar $L$ tak)
* **$W^{(l)}$**: Weight matrix for layer $l$
* **$b^{(l)}$**: Bias vector for layer $l$
* **$a^{(l-1)}$**: Previous layer ka activation output (Input layer ke liye $a^{(0)} = X$, yani input features)
* **$z^{(l)}$**: Layer $l$ ka net input (weighted sum)
* **$f(\cdot)$**: Activation function (jaise ReLU, Sigmoid, ya Softmax)
* **$a^{(l)}$**: Layer $l$ ka final output
* **$\hat{y}$**: Network ka final prediction
---
Decision Trees evaluate the optimal way to split data using impurity metrics rather than a single predictive weight equation. These core splitting formulas :

**Gini Impurity (CART Algorithm)**
$Gini=1-\sum_{i=1}^{C}(p_i)^2$

* $C$: Total number of classes.
* $p_i$: Probability of a data point belonging to class $i$ in that node.
* Measures the probability of misclassifying a randomly chosen element if it were randomly labeled according to the class distribution.

**Entropy (ID3 / C4.5 Algorithms)**
$Entropy=-\sum_{i=1}^{C}p_i\log_2(p_i)$

* $C$: Total number of classes.
* $p_i$: Probability of a data point belonging to class $i$.
* Measures the level of impurity, disorder, or uncertainty in a specific node.

**Information Gain**
$IG(S,A)=Entropy(S)-\sum_{v\in Values(A)}\frac{\vert{}S_v\vert{}}{\vert{}S\vert{}}Entropy(S_v)$

* $S$: The original dataset (parent node).
* $A$: The specific feature being evaluated for the split.
* $S_v$: The subset of $S$ where feature $A$ has value $v$.
* Calculates the reduction in entropy (or Gini impurity) after a dataset is split on a specific feature. The tree algorithm chooses the feature with the highest Information Gain for the split.

---
**Random Forest Regression (Averaging)**
$\hat{y} = \frac{1}{B} \sum_{b=1}^{B} f_b(x)$

* $B$: Total number of decision trees in the ensemble (forest).
* $f_b(x)$: The prediction of the $b$-th individual tree for input $x$.
* Averages the continuous output of all trees to reduce overall model variance and prevent overfitting.

**Random Forest Classification (Majority Voting)**
$\hat{y} = \arg\max_{c} \sum_{b=1}^{B} I(f_b(x) = c)$

* $B$: Total number of decision trees.
* $f_b(x)$: The predicted class from the $b$-th tree.
* $I(\cdot)$: Indicator function (evaluates to $1$ if the tree predicts class $c$, and $0$ otherwise).
* $c$: The specific class label being evaluated.
* Outputs the class that receives the highest number of votes across all individual trees.

**Out-of-Bag (OOB) Error Calculation**
$OOB_{error} = \frac{1}{n} \sum_{i=1}^{n} L(y_i, \hat{y}_{i, OOB})$

* $n$: Total number of training samples.
* $L$: Loss function (e.g., Mean Squared Error for regression or 0-1 loss for classification).
* $y_i$: The true target label.
* $\hat{y}_{i, OOB}$: The aggregated prediction for the $i$-th data point using *only* the subset of trees that did not include this specific data point in their bootstrap training sample.
* Provides a highly accurate internal validation metric without requiring a separate holdout validation dataset.

