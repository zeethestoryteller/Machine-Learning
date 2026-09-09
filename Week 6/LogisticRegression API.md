# 📈 Logistic Regression Classifier

### 1. Overview & Alternative Names

* **Also known as:** Logit regression, Maximum Entropy classifier (**maxent**), or Log-linear classifier.
* **Core Objective:** Minimizes the objective function:

$$\arg \min_{w, C} \text{ regularization penalty} + C \times \text{cross entropy loss}$$


* **Supported Tasks:**
  * Binary classification
  * One-vs-Rest (OVR)
  * Multinomial logistic regression



---

### 2. Implementation with Scikit-Learn (`LogisticRegression`)

#### **Basic Training Steps**

```python
from sklearn.linear_model import LogisticRegression

# Step 1: Instantiate the classifier
logit_classifier = LogisticRegression()

# Step 2: Fit the model using training data
logit_classifier.fit(X_train, y_train)

```

---

### 3. Solvers & Optimization Algorithms

Logistic regression uses specific solvers depending on the dataset size, feature scale, and multi-class requirements. By default, scikit-learn uses the **`lbfgs`** solver.

| Solver | Supported Penalties | Best Use Case / Notes |
| --- | --- | --- |
| **`lbfgs`** | `l2`, `none` | Default solver; robust for unscaled datasets. |
| **`newton-cg`** | `l2`, `none` | Handles multinomial loss; robust. |
| **`liblinear`** | `l1`, `l2` | Great for small datasets; limited to One-vs-Rest (OVR). |
| **`sag`** | `l2`, `none` | Stochastic Average Gradient; faster for large datasets. |
| **`saga`** | `elasticnet`, `l1`, `l2`, `none` | Unbiased, more flexible version of `sag`; best for large datasets with `l1` or `elasticnet`. |

---

### 4. Regularization (`penalty` & parameter `C`)

* **Types of Penalties:**
  * `l2` (Default - added for numerical stability)
  * `l1` (Sparsity / feature selection)
  * `elasticnet` (Combination of L1 and L2)
  * `none` (No penalty)


* **Parameter `C` (Inverse Regularization Strength):**
  * Must be a **positive** float.
  * Smaller `C**` $\rightarrow$ **Stronger** regularization.
  * Larger `C**` $\rightarrow$ **Weaker** regularization.



---

### 5. Handling Class Imbalance (`class_weight`)

* **Purpose:** Deals with imbalanced classes by applying differential penalties for mistakes.
* Higher values assigned to a class put a **higher emphasis/penalty** on misclassifying instances of that class.
```
model = LogisticRegression(class_weight='balanced', random_state=42)
```
---

### 6. Built-in Cross-Validation (`LogisticRegressionCV`)

* Automatically performs cross-validation to find the best hyperparameters (such as best values for **`C`** and **`l1_ratio`**) according to a specified scoring attribute.
```python
from sklearn.linear_model import LogisticRegressionCV
from sklearn.datasets import make_classification

# Sample data generate karte hain
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Step 1: Instantiate LogisticRegressionCV with ElasticNet and l1_ratios
logit_cv_elastic = LogisticRegressionCV(
    Cs=10,                      # C values ki list/number
    l1_ratios=[0.1, 0.5, 0.9],  # Try karne ke liye alag-alag l1_ratios
    cv=5,                       # Cross-validation folds
    penalty='elasticnet',       # ElasticNet penalty
    solver='saga',              # 'saga' solver zaroori hai elasticnet ke liye
    scoring='accuracy',
    random_state=42
)

# Step 2: Fit the model
logit_cv_elastic.fit(X, y)

# Results check karna
print("Best C value found:", logit_cv_elastic.C_)
print("Best l1_ratio found:", logit_cv_elastic.l1_ratio_)
```



---

### 7. Alternative: Using `SGDClassifier`

Logistic regression can also be implemented using the generic Stochastic Gradient Descent API by setting the loss parameter:

```python
from sklearn.linear_model import SGDClassifier

sgd_logistic = SGDClassifier(loss='log_loss') # (or loss='log' in older sklearn versions)
sgd_logistic.fit(X_train, y_train)

```
