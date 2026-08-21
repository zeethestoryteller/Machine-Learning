# 🗂️ Multi-Learning Classification Set Up

### 1. Overview of Multi-Learning Problems

Multi-learning problems extend classification beyond simple binary tasks into scenarios with multiple classes or outputs:

* **Multiclass Classification:** Exactly one output label per example, but the total number of unique labels is $> 2$ (e.g., Iris dataset with 3 classes, MNIST with 10 digits).
* **Multilabel Classification:** Total number of output labels is $\ge 2$, and an example can belong to multiple classes simultaneously.
* **Multi-output Classification:** Total number of output labels is $> 2$ with multiple target variables.

> *Note:* Both multilabel and multioutput models (where the number of output labels > 1) are collectively referred to as **multi-label classification models**.

---

### 2. Target Types & Formats (`type_of_target`)

You can determine the nature of your target vector using `sklearn.utils.multiclass.type_of_target`:

* **`'multiclass'`:** Contains more than two discrete values; 1D or column vector (e.g., `[1, 0, 2]` or `['apple', 'pear', 'orange']`).
* **`'multiclass-multioutput'`:** 2D array containing more than two discrete values with dimensions $> 1$.
* **`'multilabel-indicator'`:** A label indicator matrix (2D array with at least 2 columns and binary values).
* **`'binary'`:** Standard two-class classification.
* **`'continuous'` / `'continuous-multioutput'`:** Regression targets.

#### **Label Binarization Example**

```python
from sklearn.preprocessing import LabelBinarizer
import numpy as np

y = np.array(['apple', 'pear', 'apple', 'orange'])
y_dense = LabelBinarizer().fit_transform(y)
# Converts labels into a multi-class format of shape (n, k)

```

---

### 3. Multi-Class Classification Strategies (`sklearn.multiclass`)

While all scikit-learn classifiers perform multiclass classification **out-of-the-box**, you can use meta-estimators to experiment with specific multi-class strategies:

#### **A. One-vs-Rest / One-vs-All (`OneVsRestClassifier`)**

* **Strategy:** Fits one binary classifier per class $c$ (distinguishing $c$ versus all other classes).
* **Characteristics:** Computationally efficient, requires only $k$ classifiers for $k$ classes, and the resulting model is highly interpretable. Also supports multilabel classification when given an indicator matrix.

```python
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

ovr_clf = OneVsRestClassifier(LinearSVC(random_state=0))
ovr_clf.fit(X, y)

```

#### **B. One-vs-One (`OneVsOneClassifier`)**

* **Strategy:** Fits one classifier per pair of classes. Total classifiers = $\binom{k}{2}$.
* **Characteristics:** Predicts the class that receives the maximum votes (ties are broken by aggregate classification confidence). Useful when the base estimator does not scale well with the entire dataset.

```python
from sklearn.multiclass import OneVsOneClassifier

ovo_clf = OneVsOneClassifier(LinearSVC(random_state=0))
ovo_clf.fit(X, y)

```

---

### 4. Multilabel & Multi-Output Classification Strategies (`sklearn.multioutput`)

For problems with multiple target variables or outputs:

#### **A. Multi-Output Classifier (`MultiOutputClassifier`)**

* **Strategy:** Fits one independent classifier per target variable.
* **Use Case:** Predicting a series of responses from a single predictor matrix $X$.

#### **B. Classifier Chain (`ClassifierChain`)**

* **Strategy:** Arranges binary classifiers into a chain (from $0$ to $k-1$) where subsequent classifiers use the predictions of previous classifiers as input features.
* **Use Case:** Capable of **exploiting correlations among targets** in multi-label classification.
