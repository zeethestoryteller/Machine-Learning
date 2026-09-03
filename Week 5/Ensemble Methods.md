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




