# Gaussian Naive Bayes

Mathematics (Bayes' Theorem)

```
P(y|x) ∝ P(y) · Π P(xᵢ|y)   (naive: features are conditionally independent)

For continuous x: P(xᵢ|y) = Gaussian(μᵢy, σᵢy²)

```
| Variant | $P(x_i \mid y)$ | Best for |
| --- | --- | --- |
| **GaussianNB** | Gaussian | Continuous numeric |
| **BernoulliNB** | Bernoulli | Binary ($0/1$ counts) |
| **MultinomialNB** | Multinomial | Word counts (NLP) |
| **ComplementNB** | Complement | Imbalanced text |



Concept Breakdown: Naive Bayes Independence Assumption

> **Statement:** "Two dependent features do not impact GaussianNB because it calculates conditional probability independently."

> **Analysis:** This statement is FALSE. GaussianNB assumes independence. When features are actually dependent, GaussianNB double-counts their evidence, leading to overconfident predictions.

> **Tip:** MultinomialNB is specifically designed for classification with discrete count features (like word counts).

```python

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

gnb = GaussianNB(var_smoothing=1e-9)  # smoothing prevents zero variance
gnb.fit(X_train_s, y_train)

# Prior & posterior access
print(gnb.class_prior_)    # P(y=k) for each class
```
