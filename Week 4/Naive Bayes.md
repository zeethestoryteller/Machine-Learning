# Naive Bayes — Complete Guide (Theory + scikit-learn Implementation)

## 1. The Core Idea

Naive Bayes is a **probabilistic classifier** based on Bayes' theorem, with a "naive" assumption: **every feature is conditionally independent of every other feature, given the class label.**

### Bayes' Theorem
$$P(y \mid x_1, ..., x_m) = \frac{P(y) \, P(x_1, ..., x_m \mid y)}{P(x_1, ..., x_m)}$$

- $P(y)$ = **prior** — how common each class is
- $P(x_1,...,x_m \mid y)$ = **likelihood** — how likely this feature combination is, given the class
- $P(x_1,...,x_m)$ = **evidence** — same for every class, so it's just a normalizing constant (we can drop it for classification)

### The "Naive" Assumption
Computing the full joint likelihood $P(x_1,...,x_m \mid y)$ is intractable in general. Naive Bayes simplifies it by assuming every feature is independent of the others given the class:

$$P(x_i \mid y, x_1,...,x_{i-1},x_{i+1},...,x_m) = P(x_i \mid y)$$

This means the joint likelihood factorizes:

$$P(x_1,...,x_m \mid y) = \prod_{i=1}^{m} P(x_i \mid y)$$

So the classification rule becomes:

$$\hat{y} = \arg\max_y \; P(y) \prod_{i=1}^{m} P(x_i \mid y)$$

This assumption is almost never literally true (features usually do interact) — that's why it's called "naive." But it works surprisingly well in practice, especially for text classification, and it's **extremely fast** to train because you only ever estimate simple per-feature distributions instead of a huge joint distribution.

### Why it's fast
- **Training** = just estimating $P(y)$ and $P(x_i \mid y)$ for each feature — closed-form, one pass over the data, no iterative optimization.
- **Prediction** = multiply/sum a handful of probabilities per class.

This makes NB a great baseline model, especially with high-dimensional, sparse data like text (bag-of-words / TF-IDF vectors).

---

## 2. Which NB Variant to Use?

scikit-learn's `sklearn.naive_bayes` module implements 5 variants. The **only** thing that changes between them is the assumed distribution of $P(x_i \mid y)$ — the underlying Bayes logic is identical.

| Classifier | Use when... | Feature type |
|---|---|---|
| `GaussianNB` | Features are continuous, roughly bell-shaped | Numerical (real-valued) |
| `MultinomialNB` | Features are counts (e.g. word frequencies) | Non-negative integers |
| `ComplementNB` | Same as Multinomial, but classes are **imbalanced** | Non-negative integers |
| `BernoulliNB` | Features are binary (present/absent) | 0/1 boolean |
| `CategoricalNB` | Features are categorical (multiple discrete levels) | Categorical/discrete |

All five share the same API:
```python
model.fit(X_train, y_train)      # estimate parameters
model.predict(X_test)            # hard class predictions
model.predict_proba(X_test)      # class probabilities
model.predict_log_proba(X_test)  # log class probabilities
model.score(X_test, y_test)      # accuracy
```

---

## 3. GaussianNB — Continuous / Numerical Data

**Assumption:** for each class $y$, each feature $x_i$ follows a Normal (Gaussian) distribution:

$$P(x_i \mid y) = \frac{1}{\sqrt{2\pi\sigma_y^2}} \exp\left(-\frac{(x_i - \mu_y)^2}{2\sigma_y^2}\right)$$

The model just estimates the mean $\mu_y$ and variance $\sigma_y^2$ of each feature, per class.

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)
print("Accuracy:", gnb.score(X_test, y_test))
print("Class probabilities:\n", gnb.predict_proba(X_test)[:5])
```

### Parameters
| Parameter | Default | Meaning |
|---|---|---|
| `priors` | `None` | Manually set class priors $P(y)$ instead of estimating from data. Array of shape `(n_classes,)`. |
| `var_smoothing` | `1e-9` | A small value added to every feature's variance to avoid division-by-zero / numerical instability when a feature has zero variance in some class. Increase it if you get unstable/overconfident predictions. |

### Useful fitted attributes (after `.fit()`)
- `gnb.class_prior_` → estimated $P(y)$ for each class
- `gnb.theta_` → mean of each feature per class
- `gnb.var_` → variance of each feature per class
- `gnb.classes_` → the class labels seen

---

## 4. MultinomialNB — Count Data (classic text classification)

**Assumption:** features are counts drawn from a multinomial distribution (e.g., word counts in a document). This is the standard choice for **bag-of-words** text classification.

$$P(x_i \mid y) = \frac{N_{yi} + \alpha}{N_y + \alpha n}$$

where $N_{yi}$ = count of feature $i$ in class $y$, $N_y$ = total count of all features in class $y$, $n$ = number of features, and $\alpha$ is a smoothing term (see below).

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

docs = ["I love this movie", "This film was terrible",
        "Great acting and story", "Worst movie ever", "Amazing plot"]
labels = [1, 0, 1, 0, 1]  # 1 = positive, 0 = negative

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs)          # bag-of-words counts
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=0)

mnb = MultinomialNB()
mnb.fit(X_train, y_train)
print(mnb.predict(X_test))
```

### Parameters
| Parameter | Default | Meaning |
|---|---|---|
| `alpha` | `1.0` | **Laplace/Lidstone smoothing.** Prevents zero probabilities for features never seen with a given class during training (which would otherwise zero out the whole product). `alpha=1` = Laplace smoothing; `0 < alpha < 1` = Lidstone smoothing; `alpha=0` = no smoothing (risky). |
| `force_alpha` | `True` | If `False` and `alpha` is very small, sklearn silently bumps it up to avoid numerical errors. |
| `fit_prior` | `True` | Whether to learn class priors from data. If `False`, a uniform prior is used. |
| `class_prior` | `None` | Manually specify priors instead of learning them. |

### Fitted attributes
- `mnb.feature_log_prob_` → log $P(x_i \mid y)$ for every feature/class
- `mnb.class_log_prior_` → log $P(y)$
- `mnb.feature_count_` → raw counts per feature/class

---

## 5. ComplementNB — Imbalanced Text Data

**Assumption:** same multinomial model as above, but instead of estimating $P(x_i|y)$ from documents **in** class $y$, it estimates parameters from documents **not in** class $y$ (the "complement"), then normalizes. This corrects the bias multinomial NB has toward the majority class when classes are skewed.

Per the sklearn documentation referenced in the slides: CNB regularly outperforms MNB (often by a considerable margin) on text classification tasks — it's frequently recommended as the **default** choice for text over MultinomialNB.

```python
from sklearn.naive_bayes import ComplementNB

cnb = ComplementNB()
cnb.fit(X_train, y_train)
print(cnb.predict(X_test))
```

### Parameters
| Parameter | Default | Meaning |
|---|---|---|
| `alpha` | `1.0` | Same smoothing role as in MultinomialNB. |
| `force_alpha` | `True` | Same as MultinomialNB. |
| `fit_prior` | `True` | Learn class priors from data. |
| `class_prior` | `None` | Manually specify priors. |
| `norm` | `False` | Whether to apply a second normalization of the weights (the original CNB paper's "second normalization"). Rarely changed. |

---

## 6. BernoulliNB — Binary / Boolean Features

**Assumption:** each feature is a binary indicator (0 or 1) — e.g. "does this word appear in the document at all" (as opposed to MultinomialNB's word *counts*). Explicitly penalizes the **absence** of a feature, unlike MultinomialNB which just ignores absent features.

$$P(x_i \mid y) = P(i \mid y)\, x_i + (1 - P(i \mid y))(1 - x_i)$$

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(binary=True)   # binary presence/absence, not counts
X = vectorizer.fit_transform(docs)

bnb = BernoulliNB()
bnb.fit(X_train, y_train)   # X_train should be binary features
print(bnb.predict(X_test))
```

### Parameters
| Parameter | Default | Meaning |
|---|---|---|
| `alpha` | `1.0` | Laplace/Lidstone smoothing, same role as before. |
| `force_alpha` | `True` | Same as MultinomialNB. |
| `binarize` | `0.0` | Threshold for converting input features to binary. Any value `> binarize` becomes 1, else 0. Set to `None` if your input is already binary and you don't want any thresholding. |
| `fit_prior` | `True` | Learn class priors from data. |
| `class_prior` | `None` | Manually specify priors. |

---

## 7. CategoricalNB — Categorical Features (multiple discrete levels)

**Assumption:** each feature $i$ has its **own** categorical distribution (not just binary — it can take on many discrete values, e.g. "color": red/blue/green). Good fit for tabular data with nominal categorical columns (like the classic PlayTennis / weather dataset).

```python
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

# Example: categorical features must be encoded as non-negative integers
X_raw = np.array([
    ["Sunny", "Hot"], ["Overcast", "Hot"], ["Rainy", "Mild"],
    ["Sunny", "Cool"], ["Overcast", "Cool"]
])
y = [0, 1, 1, 0, 1]  # e.g. 0 = don't play, 1 = play

encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(X_raw)

canb = CategoricalNB()
canb.fit(X_encoded, y)
print(canb.predict(X_encoded))
```

### Parameters
| Parameter | Default | Meaning |
|---|---|---|
| `alpha` | `1.0` | Additive (Laplace/Lidstone) smoothing. |
| `force_alpha` | `True` | Same as above. |
| `fit_prior` | `True` | Learn class priors from data. |
| `class_prior` | `None` | Manually specify priors. |
| `min_categories` | `None` | Minimum number of categories to expect per feature (int, array, or `None`). Useful if the test set might contain a category not seen at training time — it reserves probability mass for unseen categories. |

**Important:** `CategoricalNB` expects features encoded as integers starting from 0 (use `OrdinalEncoder`, not `OneHotEncoder`).

---

## 8. Common Workflow (applies to all variants)

```python
from sklearn.naive_bayes import GaussianNB   # swap in whichever variant fits your data
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 1. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Instantiate estimator
model = GaussianNB()

# 3. Fit (estimates priors + per-feature likelihood parameters)
model.fit(X_train, y_train)

# 4. Predict
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# 5. Evaluate
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

---

## 9. Quick Decision Cheat-Sheet

- **Numeric/continuous features (e.g. sensor readings, measurements)** → `GaussianNB`
- **Word counts / term frequencies (text)** → `MultinomialNB`
- **Word counts, but classes are imbalanced** → `ComplementNB`
- **Binary features (word present/absent, yes/no flags)** → `BernoulliNB`
- **Categorical/nominal features with multiple levels** → `CategoricalNB`

## 10. Key Things to Remember When Implementing

1. **Smoothing (`alpha`) matters a lot** for Multinomial/Complement/Bernoulli/Categorical — without it, any feature value unseen during training gives that class a probability of exactly 0, wiping out the whole prediction regardless of other evidence. Tune `alpha` (e.g. via `GridSearchCV`) rather than leaving it at the default blindly.
2. **`var_smoothing` in GaussianNB** matters if features have near-zero variance — increase it if you see instability.
3. **Feature independence is an assumption, not a fact** — NB can still work well even when features are correlated, but if performance is disappointing, that's usually why.
4. **Priors**: by default all variants learn `P(y)` from class frequency in the training data (`fit_prior=True`). Set `fit_prior=False` for a uniform prior, or pass `class_prior`/`priors` explicitly if you have domain knowledge about true class proportions (e.g. in medical diagnosis where training data doesn't reflect real-world prevalence).
5. Naive Bayes gives you **probability estimates** (`predict_proba`), not just hard labels — useful for ranking/thresholding decisions.
