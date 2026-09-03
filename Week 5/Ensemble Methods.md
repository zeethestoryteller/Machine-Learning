#  Ensemble Methods : Theory

The Wisdom of Crowds
A single model makes errors. Multiple diverse models average them out.

> Two key conditions for ensembles to work:
> 1. Each model is better than random (> 50% accuracy)
> 2. Models make DIFFERENT errors (diversity)
> 
> 

Three Ensemble Strategies

| Strategy | Method | Key models |
| --- | --- | --- |
| Bagging | Train parallel on random subsets | Random Forest |
| Boosting | Train sequentially, correct errors | AdaBoost, Gradient Boosting, XGBoost |
| Stacking | Meta-learner trained on base predictions | StackingClassifier |

---
# Bagging & Random Forest

### Bagging (Bootstrap Aggregating)

>For each of `n_estimators`:
>
>1. Draw a bootstrap sample (random rows WITH replacement)
>2. Train a full Decision Tree on the sample
>
>Predict by majority vote (classification) or mean (regression)

### Random Forest : Out-of-Bag Samples

* Each tree sees ~63.2% of training samples (bootstrap).
* The remaining ~36.8% are "out-of-bag" : used as free validation.
* `oob_score_` $\approx$ 5-fold CV accuracy with much less computation.

> **Concept Breakdown: The max_features Trap in Random Forest**
> * **Problem:** What happens if you INCREASE `max_features`?
> * **Reasoning:** Each tree considers more features at each split. The trees become MORE correlated with each other (they all use the same strong features).
> * **Result:** Less diversity among trees means the **ensemble variance increases** (you lose the bagging benefit).
> * **Tip:** If `max_features = None` (all features), Random Forest degrades toward an ensemble of highly correlated deep trees.

---
# AdaBoost

### How AdaBoost works

1. Train weak learner on training data (weight all samples equally)
2. Increase weights of misclassified samples
3. Train next learner on reweighted data
4. Combine all learners with weighted voting
Repeat for `n_estimators` rounds

```python
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # "stump"
    n_estimators=200,
    learning_rate=0.5,  # shrinks contribution of each tree
    algorithm='SAMME',   # 'SAMME.R' uses probabilities (usually better)
    random_state=42
)
ada.fit(X_train, y_train)

# Staged predictions (see how score improves with more trees)
staged_scores = list(ada.staged_score(X_test, y_test))

```

---

# Gradient Boosting

### Key Idea

Each new tree fits the RESIDUALS (pseudo-residuals) of the previous ensemble.
This is gradient descent in function space.

<img width="755" height="145" alt="image" src="https://github.com/user-attachments/assets/c4db3244-5260-455d-b83d-fc177b44e7ec" />


```python
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,  # small is better, need more trees
    max_depth=3,         # shallow trees are better for boosting
    subsample=0.8,       # Stochastic GB : reduces overfitting
    min_samples_split=10,
    max_features='sqrt',
    validation_fraction=0.1,
    n_iter_no_change=15,  # early stopping
    random_state=42
)
gb.fit(X_train, y_train)
print(f"Best iteration: {gb.n_estimators_}")  # with early stopping

```

### Gradient Boosting Hyperparameter Guide

| Parameter | Effect of increasing | Typical range |
| --- | --- | --- |
| n_estimators | More trees, can overfit | 100–5000 |
| learning_rate | Faster but coarser | 0.01–0.3 |
| max_depth | More complex trees | 2–6 |
| subsample | More randomness | 0.5–1.0 |
| min_samples_leaf | Less overfit | 1–50 |

*Classic tradeoff:* Low `learning_rate` + high `n_estimators` = best accuracy (with early stopping)

---

# Voting & Stacking Classifiers

```python
from sklearn.ensemble import VotingClassifier, StackingClassifier

# Voting : equal vote
voting = VotingClassifier([
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('lr', LogisticRegression(max_iter=1000))
], voting='soft')  # 'hard' = majority, 'soft' = avg probabilities

# Stacking : meta-learner
stacking = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('gb', GradientBoostingClassifier(n_estimators=100))
    ],
    final_estimator=LogisticRegression(),
    cv=5,
    stack_method='predict_proba'
)

```

---

**Key Takeaways**

| Model | Bias | Variance | Interpretable | Fast |
| --- | --- | --- | --- | --- |
| Decision Tree | Low | High | ✓ | ✓ |
| Bagging | Medium | Low | ✗ | ✓ |
| Random Forest | Low | Low | Partial | ✓ |
| AdaBoost | Low | Medium | ✗ | ✓ |
| Gradient Boosting | Very Low | Medium | ✗ | Slower |

