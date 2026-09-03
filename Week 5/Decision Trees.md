# Decision Trees

### How a Decision Tree splits
```
For each candidate (feature, threshold):
    Split data into LEFT and RIGHT subsets
    Compute impurity of each subset
    Choose the split that minimises weighted impurity

Repeat recursively until stopping criteria
```
>*Pen & Paper Example: Manual Decision Tree Splitting* *Input:* *A dataset with features Age and Salary, and a binary target (Buy=0 or 1).*
>
>*1. Candidate Splits:* *Sort the dataset by Age. The midpoints between consecutive ages become candidate thresholds (e.g., Age > 25, Age > 30). Do the same for Salary.*
>
>*2. Evaluate Gini:* *For a split like "Age > 25", calculate the Gini impurity of the Left child (Age <= 25) and Right child (Age > 25).*
>
>*3. Weighted Gini:* *Combine them using* *(n_Left / total) * Gini_Left + (n_Right / total) * Gini_Right* *.*
>
>*4. Decision:* *Calculate the weighted Gini for EVERY candidate threshold across ALL features. The one with the lowest weighted Gini is chosen as the root node.*
>
>*5. Prediction:* *To predict a new test point, traverse the tree based on its feature values. If it lands in a leaf with 3 'Yes' and 1 'No', the prediction is the majority class: 'Yes' (with 75% probability).*

### Concept Breakdown: When can a node split?* *A node must satisfy ALL criteria to split:*

1. Number of samples >= min_samples_split
2. Each resulting child would have >= min_samples_leaf
3. Depth < max_depth
4. max_leaf_nodes not reached
5. Impurity decrease >= min_impurity_decrease *Tip:* If min_samples_split=7 and min_samples_leaf=4 , a node with 10 samples can split, but only if the children get sizes like [4,6], [5,5], or [6,4].

### Impurity Measures

| Criterion | Formula | Use | Notes |
| --- | --- | --- | --- |
| Gini (classification) | $1 - \sum p_k^2$ | Default, fast | Measures probability of misclassification |
| Entropy (classification) | $-\sum p_k \log_2(p_k)$ | Information gain | More computationally expensive |
| MSE (regression) | $\sum(y_i - \bar{y})^2 / n$ | Default for regression | Minimises variance |

### Gini vs Entropy Example
```
Node : 50 samples , 40 class A , 10 class B
Gini = $1 - (40/50)^2 - (10/50)^2 = 1 - 0.64 - 0.04 = 0.32$
Entropy = $-(0.8)\log_2(0.8) - (0.2)\log_2(0.2) = 0.259 + 0.464 = 0.72$
```
### Hyperparameter Guide for Decision Trees

| Parameter | Effect of increasing | Typical range |
| --- | --- | --- |
| max_depth | More complex, risk overfit | 3–15 |
| min_samples_split | Simpler tree | 2–50 |
| min_samples_leaf | Simpler tree, stable leaves | 1–20 |
| ccp_alpha | More pruning | 0.0001–0.01 |

```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree, export_text
import matplotlib.pyplot as plt

dt = DecisionTreeClassifier(
    criterion='gini',      # 'gini' or 'entropy'
    max_depth=5,           # limits tree depth (most important regulariser)
    min_samples_split=20,  # minimum samples to split a node
    min_samples_leaf=10,   # minimum samples in leaf
    max_features=None,     # 'sqrt', 'log2', int, float
    ccp_alpha=0.0,         # pruning parameter (larger → more pruned)
    class_weight='balanced',
    random_state=42
)

dt.fit(X_train, y_train)

# Visualise
plt.figure(figsize=(20,10))
plot_tree(dt, filled=True, feature_names=X.columns, class_names=['0','1'], max_depth=3)
plt.savefig('tree.png', dpi=100, bbox_inches='tight')

# Text export
print(export_text(dt, feature_names=list(X.columns)))

# Feature importance
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': dt.feature_importances_
}).sort_values('importance', ascending=False)
print(importance_df)

```


