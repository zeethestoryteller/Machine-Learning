# Machine Learning Practice — Complete Study Guide
*A simple explanation of every concept tested in your practice exams*

---

## 1. Data Preprocessing & Feature Engineering

### 1.1 Scaling (StandardScaler, MinMaxScaler, MaxAbsScaler)

**Why scale?** Many algorithms (KNN, KMeans, SVM, gradient-descent-based models) compare distances between numbers. If one feature ranges 0–1 and another ranges 0–1,000,000, the big one dominates. Scaling puts everything on a level playing field.

- **StandardScaler**: transforms each column so mean = 0, std = 1.
  Formula: `(x - mean) / std`
  - `.fit(X)` learns mean and std from training data.
  - `.transform(X_new)` applies the **same** mean/std learned earlier — it does NOT recalculate them from X_new.
  - **Key exam trap**: when you call `scaler.transform(X_new)` on new data, it uses the ORIGINAL fit's mean/std, not new data's mean/std.

  Example: `X = [2,6,10,14]` → mean=8, std=√[(36+4+4+36)/4]=√20≈4.47
  For `X_new=[8]`: `(8-8)/4.47 = 0`

- **MinMaxScaler**: squishes data into a range, default [0,1].
  Formula: `(x - min) / (max - min)`
  For `arr=[10,20,30,40,50]`, min=10, max=50.
  `X_new=8`: `(8-10)/(50-10) = -0.05` (note: can go negative/above 1 if X_new is outside original min/max!)

- **MaxAbsScaler**: divides each column by the maximum absolute value in that column. Keeps sign, scales to [-1,1]. Good for sparse data.
  For column `[1,2,0]`, max abs = 2 → transformed: `[0.5, 1, 0]`

**How to solve these questions:** Always compute statistics (mean/std or min/max) from the **training** array first, then apply the formula to the **test** array using those training statistics.

### 1.2 Encoding (turning categories into numbers)

- **LabelEncoder**: converts categories into integers alphabetically.
  `['apple','banana','cherry']` → apple=0, banana=1, cherry=2 (sorted alphabetically, then assigned in order of first appearance... actually it's sorted alphabetically and assigned 0,1,2...).
  Count how many times each label appears to answer "how many zeros" type questions.

- **OneHotEncoder**: creates a new binary column for each category.
  `drop='first'` removes the first category's column (avoids redundancy — if you know all other columns are 0, you know it's the dropped category).
  Example: Gender ['M','F','M'] with drop='first' → F becomes column, M becomes reference: [1],[0],[1] (or similar, alphabetical order first).

- **MultiLabelBinarizer**: used when each row can have MULTIPLE labels at once (not just one class). Each unique tag becomes a column; 1 if present, 0 if absent.
  `classes=['cold','hot','large','small']`
  Row `['large','cold']` → looking up position of 'cold'(0) and 'large'(2) → `[1,0,1,0]`

**Tip:** When you see `y_train` as a list of lists (multiple tags per row), that's your clue it's MultiLabelBinarizer, not LabelEncoder.

### 1.3 Handling Missing Values (Imputation)

- **SimpleImputer(strategy='mean')**: replaces missing values in a column with that column's mean (computed from the non-missing values in that SAME column).
  Example: F3 column = [9,7,?,5]. Mean of known values (9+7+5)/3 = 7. So ? = 7.

- **SimpleImputer(strategy='median')**: same idea but uses the median.

- **KNNImputer(n_neighbors=k)**: replaces a missing value by looking at the k most similar ROWS (based on other columns) and averaging their value in the missing column.
  You need to compute distances between rows using the other features, find the k nearest, then average their target column value.

- **groupby().transform()** for imputation: replaces missing values with the mean/median of that row's **group** (e.g., fill missing Age with the mean Age of that person's Department), not the whole column.
  Code pattern: `df['Age'] = df.groupby('Department')['Age'].transform(lambda x: x.fillna(x.mean()))`

### 1.4 ColumnTransformer

Applies **different transformations to different columns** at once.
```python
ColumnTransformer([
    ('num', StandardScaler(), ['Height','Weight']),
    ('cat', OneHotEncoder(drop='first'), ['City'])
])
```
**To find output shape:** count columns from each transformer and add them up.
- StandardScaler on 2 numeric columns → 2 columns (unchanged count)
- OneHotEncoder with drop='first' on a categorical column with `n` unique values → `n-1` columns

So if `City` has 3 unique values (A,B,C) and drop='first', OneHotEncoder gives 2 columns. Total = 2 (numeric) + 2 (encoded) = 4 columns. Rows stay the same as input rows.

### 1.5 FeatureUnion

Combines the **output** of several transformers **side by side** (concatenates their columns), unlike Pipeline which runs steps in sequence.
`FeatureUnion([('poly', PolynomialFeatures), ('pca', PCA)])` → output columns = poly's output columns + PCA's output columns.

### 1.6 PolynomialFeatures

Creates new features by multiplying existing ones together, up to a given degree.
- `degree=2` on `[x1, x2]` gives: `[1, x1, x2, x1², x1*x2, x2²]` (the 1 is the bias/intercept term).
- `include_bias=False` removes the leading `1`.
- `interaction_only=True` removes the squared terms (x1², x2²), keeping only products of DIFFERENT features: `[1, x1, x2, x1*x2]`.

>**Counting output features formula:** for `n` input features and degree `d`, the count follows combinations with repetition:
> Fow all features: C(n+d, d) = (n+d)! / n!d!
>
>When include_bias = flase: C(n+d, d) - 1
>
>
or (n + d) / d. But easiest is to just work it out by hand for small n like the exam does.

---

## 2. Regression Algorithms

### 2.0 DummyRegressor:
`strategy= 'median' , 'mean', 'quantile', 'constant'`
it calculates value from target column then use it as a prediction for every new point.


### 2.1 Linear Regression

Fits a straight line/plane: `y = w0 + w1*x1 + w2*x2 + ...`
- `.fit(X,y)` learns the weights (coefficients) that minimize squared error.
- `.coef_` gives the weights, `.intercept_` gives w0.
- `fit_intercept=False` forces the line through the origin (no w0 term).
- `.predict(X_new)` plugs numbers into the learned equation.

**Solving trick:** If data is created by an exact formula (like `y = X·[1,1] - 1`), Linear Regression will learn that EXACT formula perfectly (since there's no noise), so you can just compute the formula on the test point yourself instead of "simulating" the model.

### 2.2 Ridge & Lasso Regression (Regularized Linear Models)

Both add a **penalty** to keep the weights (coefficients) small, which reduces overfitting.

> **Ridge (L2 penalty)**: penalizes the *sum of squares* of coefficients. Shrinks coefficients toward zero but rarely makes them exactly zero. Best when all features are usfull.

> **Lasso (L1 penalty)**: penalizes the *sum of absolute values* of coefficients. CAN shrink coefficients all the way to exactly zero — this makes Lasso good for feature selection. best when few features are usfull

> **ElasticNet**: Usfull when features are co-related.
> 
> Use belend of L1 + L2
>
> MSE + λ(ρ·L1 + (1−ρ)·L2)
>
> l1 ratio = 0 ridge, l1 ratio = 1 lasso

**The `alpha` parameter** controls how strong the penalty is:
- alpha = 0 → behaves just like plain Linear Regression (no penalty).
- Higher alpha → stronger penalty → smaller coefficients → simpler model (more bias, less variance, may underfit).
- Lower alpha → weaker penalty → coefficients closer to normal Linear Regression → may overfit.

**Common exam logic**: comparing two Lasso models with different alphas —
- Higher alpha model will have SMALLER (or more zero) coefficients.
- Higher alpha does NOT guarantee better accuracy on unseen data (that depends on the data) — this is usually the FALSE option.
- Lower alpha model may fit training data almost perfectly but that doesn't mean it exactly matches y unless alpha=0 removes all penalty.

### 2.3 SGDRegressor

Trains a linear model using Stochastic Gradient Descent (updates weights bit by bit using small batches/single samples), instead of solving the equation exactly.
- `shuffle=True` (default) shuffles the training data before each epoch — this is how you "shuffle training data after each epoch."
- Comes from `sklearn.linear_model`, NOT `sklearn.preprocessing`.
- `learning_rate` and `eta0` control step size.

### 2.4 Decision Trees (Regressor/Classifier)

A tree that splits data into branches based on feature values to make predictions.

**Key parameters:**
- `max_depth`: maximum number of levels the tree can grow. Lower max_depth = simpler tree = usually LOWER accuracy on training data (or same), never guaranteed to increase.
- `min_samples_split`: minimum number of samples a node MUST have to be considered for splitting.
- `min_samples_leaf`: minimum number of samples that must exist in EACH child after a split.
- `max_leaf_nodes` : limits the maximum number of leaf nodes (terminal nodes where predictions are made) that a Decision Tree can grow.
-**`ccp_alpha`** (Cost-Complexity Pruning Alpha) controls the pruning penalty for a Decision Tree.

  * **How it works:** Instead of stopping tree growth early (like `max_depth` or `max_leaf_nodes`), the algorithm first grows a fully overfitted tree and then prunes back branches. It balances tree size against error using the cost-complexity formula: $R_\alpha(T) = R(T) + \alpha \vert{}T\vert{}$ (where $R(T)$ is training error, $\vert{}T\vert{}$ is the number of leaf nodes, and $\alpha$ is `ccp_alpha`).
  * **Effect of alpha:**
  * **`ccp_alpha = 0` (default):** No pruning is performed, allowing the tree to grow to its maximum possible size (fully fitted to the training data, prone to overfitting).
  * **Higher `ccp_alpha`:** Imposes a heavier penalty for having more leaf nodes, resulting in more aggressive pruning, a simpler tree, higher bias, and lower variance.

**Splitting rule (very testable):** A split is allowed ONLY IF:
1. The node has ≥ `min_samples_split` samples, AND
2. BOTH resulting child nodes have ≥ `min_samples_leaf` samples each.

Example: `min_samples_split=6, min_samples_leaf=4`. Node has 12 samples, splits into 3 and 9.
- Node has 12 ≥ 6 ✓ but left child has only 3 < 4 ✗ → split NOT allowed.

**Decreasing max_depth effect:** training score will "decrease or stay the same" (never increase) because a shallower tree can only be equally or less flexible.

>Notes:
>if tree is balanced then maximum number of leaf nodes at a given depth $d$ is calculated: 2^d
>
>maximum possible total number of nodes—both internal and leaf nodes: $2^{d+1} - 1$
>
>tree cannot have more leaf nodes than total training samples
>
>if the tree is not balanced then the max possible no of leafe nodes can not be more than $2^{d}$
>
>the number of leaf nodes in a pruned tree will be less than or equal to max limit $2^{d}$

### 2.5 Random Forest

An ensemble of many Decision Trees, each trained on a random subset of data (bagging) and a random subset of features at each split.

**Key parameters:**
- `n_estimators`: number of trees.
- `max_features`: number of features considered at EACH split, for EVERY tree (not "only some trees" — this applies to every split in every tree).
- `bootstrap=True`: each tree is trained on a random sample of rows drawn WITH replacement (bootstrap sample), not the entire dataset unshuffled.
- `max_depth`: max depth per tree.

**Common trap answer:** "Each tree will consider only `max_features` randomly chosen features when looking for the best split at every node" — this is the TRUE statement (applies per split, per tree, always).

### 2.6 DummyRegressor

A baseline model that ignores input features entirely.
- `strategy='mean'`: always predicts the mean of the training y values, no matter what X_test is.
- `strategy='median'`: always predicts the median.
Useful to compare — if your real model isn't better than DummyRegressor, something's wrong.

Example: y_train=[10,15,20,25,30], mean=20. So DummyRegressor predicts 20 for EVERY test point, regardless of X_test values.

---

## 3. Classification Algorithms

### 3.1 K-Nearest Neighbors (KNN)

Classifies a new point by looking at its `k` closest training points (by distance) and taking a majority vote (for classification) or average (for regression).

> #Note
> $k$-Nearest Neighbors ($k$-NN) is a lazy learning algorithm (or instance-based learning algorithm).
> It does not build an explicit generalized model during the training phase; it simply stores the training dataset.
> when a prediction is required for a new query point, the model must compute the distance from the query point to all stored training samples, sort them, and select the $k$-nearest neighbors. This makes prediction time computationally expensive, creating a bottleneck.

**Steps to solve by hand:**
1. Compute distance (usually Euclidean) from test point to every training point.
2. Pick the `k` smallest distances.
3. Look at the labels of those `k` points.
4. Majority class wins.

Example: X_train points and labels given, test point [3,2], k=3 → find 3 nearest, take majority label.

**Radius-based variant (RadiusNeighborsClassifier)**: instead of picking a FIXED number of neighbors, it picks ALL points within a given `radius` distance, then votes among those. Different from KNN's fixed k.

### 3.2 Logistic Regression

Despite the name, it's a CLASSIFICATION algorithm (predicts probability of a class), not regression.
Uses a sigmoid/logistic function to squash outputs into [0,1] representing probability.

**Solvers**: `liblinear`, `lbfgs`, `sag`, `saga` are different optimization algorithms.
- **`liblinear` does NOT support true multinomial (softmax) approach for multi-class problems** — it uses one-vs-rest instead. The others (`lbfgs`, `sag`, `saga`, `newton-cg`) DO support genuine multinomial handling.
- This is the key fact for "which solver CANNOT handle multinomial multiclass" — answer: `liblinear`.

### 3.3 Perceptron

A simple linear classifier, an early neural network unit.
- By default, calling `.fit()` again RETRAINS from scratch, discarding old learned weights.
- `warm_start=True`: makes subsequent `.fit()` calls CONTINUE training from the previously learned weights instead of resetting.

### 3.4 MLPClassifier / MLPRegressor (Multi-Layer Perceptron / Neural Network)

A neural network with one or more "hidden layers" of neurons.   
> Scaling the data using MinMaxScaler is ecential.

**`hidden_layer_sizes=(50,30)`** means: TWO hidden layers, first with 50 neurons, second with 30 neurons. (NOT features/outputs/epochs — purely the architecture of hidden layers.)

**`activation`** parameter (e.g., 'relu', 'logistic', 'tanh') sets the activation function used in HIDDEN layers.

**`out_activation_`** (an attribute you check AFTER fitting) tells you what activation was actually used on the OUTPUT layer:
- Binary classification → typically 'logistic' (sigmoid)
- Multi-class classification → 'softmax'
- Regression → 'identity' (i.e., no transformation, raw linear output)

So if you load Iris (3 classes) and fit MLPClassifier, `rs.out_activation_` = 'softmax' (multi-class), NOT whatever `activation` you set for hidden layers.

**`n_layers_`** = number of hidden layers + input layer + output layer (total layers in the network).
Example: `hidden_layer_sizes=(12,15,13,11,12,8)` → 6 hidden layers + 1 input + 1 output = 8 total layers.

**`coefs_[i].shape`**: `coefs_` is a list of weight matrices, one between each pair of consecutive layers. `coefs_[i]` has shape `(neurons_in_layer_i, neurons_in_layer_i+1)`.

### 3.5 Naive Bayes Variants

All based on Bayes' theorem but differ in what kind of data they assume:

- **GaussianNB**: for CONTINUOUS numeric features assumed to follow a normal (bell-curve) distribution. Use when features are things like height, weight, age (continuous numbers).
> The prior probability  $p(y = \text{"Yes"})$ = proportion of that class within the entire training dataset
- **MultinomialNB**: for COUNT data (e.g., word counts in text).
- **BernoulliNB**: for BINARY features (0/1, present/absent).
- **CategoricalNB**: for CATEGORICAL features with multiple discrete categories (not ordered numbers).

**How to decide**: look at the feature columns. If they're continuous numbers (Height, Weight, Age) → GaussianNB.

### 3.6 SVC (Support Vector Classifier)

Finds the best boundary (hyperplane) that separates classes with maximum margin.
- `kernel='linear'`: straight-line boundary.
- `gamma`: controls influence of individual points (for non-linear kernels).
- `.support_vectors_`: the actual data points closest to the decision boundary — the "hardest" points to classify, which define the boundary. To find these, you'd need to know which training points end up as support vectors after fitting (usually points near the boundary between classes, not points deep inside one class).

---

## 4. Ensemble Methods (Combining Models)

### 4.1 Bagging (Bootstrap Aggregating)

Trains multiple copies of the SAME algorithm on different random (bootstrap = with replacement) samples of the data, then averages/votes their predictions.
- **Goal**: reduce VARIANCE (overfitting), not bias.
- Example: RandomForest is bagging of Decision Trees.
- `BaggingClassifier(base_estimator, n_estimators, max_samples, bootstrap)`
  - `bootstrap=True`: samples WITH replacement (default).
  - `bootstrap=False`: samples WITHOUT replacement.
``` 
BaggingClassifier/Regressor
   → Same model TYPE, different bootstrap SAMPLES of rows (and optionally columns).
   → Reduces VARIANCE of a single high-variance base learner.

RandomForestClassifier/Regressor
   → BaggingClassifier/Regressor, base learner fixed to DecisionTree,
     PLUS random feature subsampling at every individual split (not just per tree).
   → Reduces variance even further than plain bagged trees, by decorrelating the trees.
   → Setting lower to max_features: It increases the variance among individual trees.
```
### 4.2 Boosting

Trains models SEQUENTIALLY, where each new model tries to fix the mistakes of the previous ones (focuses more on misclassified points).
- **Goal**: reduce BIAS (underfitting) by sequentially improving.
- Example: AdaBoost.
- Models are NOT trained independently/in parallel (that's bagging) — they're trained one after another, each depending on the previous one's errors.

**AdaBoostClassifier parameters:**
- `base_estimator`: the weak learner used at each step (often a shallow Decision Tree).
- `n_estimators`: number of boosting rounds.
- `learning_rate`: shrinks the contribution of each classifier; doesn't necessarily "always reduce training accuracy" — that's a trap answer (it's more nuanced, affects convergence speed, not a guaranteed accuracy decrease).
- Shallower base trees (e.g., max_depth=1, "stumps") are LESS likely to overfit than deeper base trees (max_depth=3) — because deeper individual trees are more complex/expressive.

**Common false statement traps:**
- "Increasing n_estimators ALWAYS improves test accuracy" → FALSE (can overfit after some point).
- "Boosting trains all models independently in parallel" → FALSE (it's sequential).
- "Bagging reduces bias" → FALSE (bagging reduces variance; boosting reduces bias).

### 4.3 VotingClassifier

Combines predictions from several DIFFERENT algorithms.
- `voting='hard'`: takes the majority class label (simple vote count).
- `voting='soft'`: AVERAGES the predicted class PROBABILITIES from each model, then picks the class with highest average probability. (Needs each base model to support `predict_proba`.)
- `flatten_transform=True` (with soft voting) affects the shape of `.transform()` output (flattens per-classifier probabilities into one row per sample) — doesn't affect prediction accuracy itself.

---

## 5. Unsupervised Learning (Clustering & Dimensionality Reduction)

### 5.1 K-Means Clustering

Groups data into `k` clusters by:
1. Randomly placing `k` centroids.
2. Assigning each point to its nearest centroid.
3. Recomputing centroids as the mean of assigned points.
4. Repeat until stable (convergence).

**Key attributes:**
- `.labels_`: the cluster index (0, 1, 2, ...) assigned to EACH data point (not the feature values, not centroids).
- `.cluster_centers_`: coordinates of the centroids.
- `.inertia_`: sum of squared distances of points to their assigned centroid (lower = tighter clusters) — used in the Elbow Method.

**Elbow Method**: plot inertia vs number of clusters (k). The "elbow" (where the curve bends and flattens) suggests the best k. In the exam's graph, the curve flattens around k=5, so the correct `centers=...` value used to generate that data would be 5.

**Important nuance:** K-Means does NOT guarantee equal-sized clusters — even with a fixed number of "centers" in make_blobs, the resulting cluster sizes (after fitting K-Means or just grouping by label) can vary; they are NOT necessarily equal.

**n_init**: controls how many times K-Means runs with different random centroid seeds, keeping the best result (based on lowest inertia) — this is what "number of clusters is a hyperparameter validated via elbow/silhouette" refers to conceptually (number of CLUSTERS is chosen via elbow/silhouette, separate from n_init).

### 5.2 Hierarchical Agglomerative Clustering

- Starts with EVERY point as its OWN cluster (not "all points in one cluster" — that's the opposite/divisive approach).
- REPEATEDLY MERGES the two closest clusters, one merge at a time, until you reach one cluster (or a stopping condition).
- Does NOT require specifying number of clusters beforehand (unlike K-Means) — you can cut the dendrogram at any level afterward.
- Does NOT update centroids (that's a K-Means concept).

**Dendrogram**: a tree diagram showing merge order and distance. Lower merge height = more similar. To find "which sample is most similar to sample labeled 1," find which sample joins with sample 1 at the LOWEST height in the dendrogram (they merge first = most similar).

### 5.3 PCA (Principal Component Analysis)

Reduces the number of features while keeping as much "spread"/variance in the data as possible. Finds new axes (principal components) that capture the most variance.

- `n_components`: how many new dimensions to keep.
- `.explained_variance_ratio_`: what fraction of total variance each component explains. If data is perfectly along ONE direction (e.g., points on a line `[[1,1],[2,2]]`), the FIRST component explains 100% (ratio=1.0) and the second explains 0% (ratio=0.0) — since there's zero spread in the perpendicular direction.

### 5.4 SVD (Singular Value Decomposition)

Used on a user-item matrix (like in recommender systems) to REDUCE DIMENSIONALITY — compressing the huge sparse matrix into smaller latent factor representations, uncovering hidden patterns (like "genres" users implicitly like) without needing labeled data.

### 5.5 make_blobs

A function to generate synthetic clustered data for testing clustering algorithms.
`make_blobs(n_samples, centers, n_features, cluster_std, random_state)`
- `centers`: number of blob clusters to generate (or specific coordinates).
- Even with a fixed `n_samples` and `centers`, the samples are NOT necessarily split perfectly evenly among clusters — sizes can vary slightly. (This is a repeated trap in the exams.)

---

## 6. Model Evaluation Metrics

### 6.1 Confusion Matrix

A table comparing actual vs predicted labels.
```
                Predicted
              0    1    2
Actual  0   TN   FP   ..
        1   FN   TP   ..
        2   ..   ..   ..
```
`confusion_matrix(y_true, y_pred, labels=[...])` — the `labels` parameter controls ROW/COLUMN ORDER. Always check the labels order given!

**How to build it by hand:** For each (actual, predicted) pair, increment the count at [row=actual, column=predicted].

### 6.2 Precision, Recall, F1-Score

For a SPECIFIC class (say class "1"):
- **True Positive (TP)**: predicted=1 AND actual=1
- **False Positive (FP)**: predicted=1 AND actual≠1
- **False Negative (FN)**: predicted≠1 AND actual=1

- **Precision** = TP / (TP + FP) → "Of everything I PREDICTED as this class, how many were actually right?"
- **Recall** = TP / (TP + FN) → "Of everything that ACTUALLY IS this class, how many did I catch?"
- **F1-score** = 2 × (Precision × Recall) / (Precision + Recall) → harmonic mean, balances both.

**Worked example:**
`y_true = [1,0,1,1,0,1]`, `y_pred = [1,0,0,1,0,1]`
For class 1: TP = positions where both are 1 → indices 0,3,5 → TP=3
FN = actual 1 but predicted 0 → index 2 → FN=1
FP = actual 0 but predicted 1 → none → FP=0
Recall = 3/(3+1) = 0.75
Precision = 3/(3+0) = 1.0
F1 = 2×(1×0.75)/(1+0.75) ≈ 0.857

**Reading a custom "metric" function**: Look at what TP/FN/FP combination it computes, and match the formula to Precision/Recall/F1/Accuracy.
- If it's `TP / (TP+FN)` → Recall.
- If it's `TP / (TP+FP)` → Precision.

### 6.3 R² Score (Coefficient of Determination)

Measures how well predictions match actual values, relative to just guessing the MEAN every time.
Formula: `R² = 1 - (SS_res / SS_tot)`
where `SS_res = Σ(y_true - y_pred)²` and `SS_tot = Σ(y_true - mean(y_true))²`

**Range**: R² can go from **-∞ to 1**.
- R²=1: perfect predictions.
- R²=0: model is exactly as good as always predicting the mean.
- R²<0: model is WORSE than just guessing the mean (yes, this can happen — that's why the range is -infinity to 1, not 0 to 1).

### 6.4 Accuracy vs Precision vs Recall vs F1 — quick decision guide
- **Accuracy** = (TP+TN)/(all samples) → overall correctness, but misleading with imbalanced classes.
- Use **Recall** when missing positives is costly (e.g., disease detection — don't want false negatives).
- Use **Precision** when false alarms are costly (e.g., spam filter — don't want to mark real emails as spam).
- Use **F1** when you want a balance of both.

**Important note**: statements like "precision should always be maximized over recall" are FALSE — it depends entirely on the problem context, there's no universal rule.

### 6.5 Silhouette Score / Inertia (for clustering, not classification)

- **Silhouette Score**: measures how well-separated clusters are (ranges -1 to 1, higher is better). Valid for evaluating K-Means/clustering output.
- **Inertia**: sum of squared distances to nearest centroid (lower is better, but always decreases as k increases, so not directly comparable across different k without the elbow method).
- Note: metrics like F1, R², Negative MAE are for supervised (labeled) tasks — NOT valid ways to judge unlabeled clustering output. Only Silhouette Score and Inertia make sense for evaluating KMeans in an unsupervised setting.

### 6.6 Cosine Similarity

Measures the angle between two vectors (ignores magnitude, focuses on direction) — common for text/recommendation similarity.
Formula: `cos_sim(A,B) = (A · B) / (||A|| × ||B||)`
where `A·B` = dot product = sum of element-wise products, and `||A||` = sqrt(sum of squares of A).

**Worked example:** A=(1,2,3), B=(1,0,0)
Dot product = 1×1 + 2×0 + 3×0 = 1
||A|| = √(1+4+9) = √14 ≈ 3.742
||B|| = √(1) = 1
cos_sim = 1 / (3.742×1) ≈ 0.27

**Correct numpy code** must use dot product divided by the PRODUCT of norms — watch for trick answers that use sum() instead of dot(), or add instead of multiply.

---

## 7. Model Selection & Validation

### 7.1 train_test_split

Splits data into training and testing sets.
- `test_size`: fraction reserved for testing (e.g., 0.2 = 20% test).
- `random_state`: a "seed" number — using the SAME random_state gives the SAME split every time (reproducible). Different random_state values → different (usually) row selections.
- **Key fact**: if `random_state` is the SAME across two calls with the same test_size, the SAME rows go into X_train both times. If `random_state` differs, rows will (almost certainly) differ.
- `test_size=0.2` and `train_size=0.8` are equivalent (giving the same split), so if random_state also matches, the splits will be identical.

### 7.2 GridSearchCV

Tries **EVERY POSSIBLE COMBINATION** of hyperparameters you provide in `param_grid`, using cross-validation to score each combination, and returns the best one.

**Counting combinations**: multiply the number of choices for each parameter together.
Example: `n_estimators: [50,100,150,200]` (4 options) × `max_depth: [None,5,10,15,20]` (5 options) × `min_samples_split: [2,3,4,5,6,8,10]` (7 options)
Total = 4 × 5 × 7 = 140 combinations.

- `.best_score_`: the MEAN cross-validated score of the BEST combination (average across folds, not the max single-fold score).
- Pipeline hyperparameter naming: use `stepname__paramname` (double underscore) to refer to a specific pipeline step's parameter, e.g., `poly__degree` refers to the `degree` parameter of the step named `'poly'`.

### 7.3 RandomizedSearchCV

Similar to GridSearchCV but instead of trying EVERY combination, it randomly samples a FIXED NUMBER (`n_iter`) of combinations.
- `n_iter=4` → only 4 random combinations get tested, NOT all possible combinations (unlike GridSearchCV).
- `random_state` affects WHICH combinations get randomly chosen — so changing it CAN change results (it does have an effect, contrary to a trap answer saying it has "no effect").
- Higher `cv` (more folds) = MORE computation, not cheaper (another common trap: "changing cv to 8 will be computationally cheaper" is FALSE — more folds = more model fits = more expensive).

### 7.4 Cross-Validation

Splits data into multiple "folds," trains on some, tests on the rest, repeating so every fold gets used as test once, then averages the scores. Reduces the risk of a lucky/unlucky single split.

**LeaveOneOut (LOOCV)**: an extreme form of cross-validation where EACH fold leaves out just ONE sample for testing and trains on ALL the rest. So if you have `n` samples, LOOCV trains the model `n` TIMES (once per sample left out).
Example: 5 samples → LOOCV trains model 5 times.

**LeaveOneOut.split(X)** on data `X=[1,2,3,4]` (4 samples): produces 4 splits, each time leaving exactly one index out as test, rest as train.

---

## 8. Text Processing / NLP

### 8.1 CountVectorizer

Converts text documents into a matrix of word (or n-gram) COUNTS.
- Each unique word/n-gram = one column (feature).
- Each cell = how many times that word appears in that document.
- `.vocabulary_`: a dictionary mapping each word to its column INDEX (not count) — e.g., `{'this': 5, 'is': 2, ...}` shows word→index mapping, sorted alphabetically for index assignment.
- `ngram_range=(2,2)`: only consider 2-word phrases ("bigrams"), not single words.
- `lowercase=False`: keeps original capitalization (treats "Data" and "data" as DIFFERENT tokens).

**Computing total word count** (`.toarray().sum()`): count all words/n-grams across all documents combined.

### 8.2 TfidfVectorizer

Like CountVectorizer, but instead of raw counts, it weighs words by how important/rare they are:
`TF-IDF = Term Frequency × Inverse Document Frequency`
Words that appear in MANY documents get a LOWER weight (they're less distinctive); rare words get boosted.

- `smooth_idf=True` (default): adds 1 to document frequencies to avoid division-by-zero; means a word appearing in ALL documents does NOT get an IDF of exactly zero (it gets a small positive value, not literally zero) — that "ensures IDF equals zero" statement is FALSE.
- `max_features=1000`: keeps only the top 1000 most frequent terms.
- `min_df=3`: ignores terms that appear in FEWER than 3 documents (removes rare/noise terms).
- `.get_feature_names_out()`: returns the list of unique terms (vocabulary) used as columns — `len()` of this tells you vocabulary size.

### 8.3 Cosine Similarity (again, for text)

Same formula as Section 6.6 — very commonly used to compare TF-IDF or Count vectors of documents to see how similar their content is.

---

## 9. Recommender Systems

### 9.1 Content-Based Filtering

Recommends items based on the ITEM'S OWN FEATURES (genre, cast, director, description, etc.) and matching them to a user's known preferences.
- **Great for "cold start" items** (new items with no ratings yet) because it doesn't need any user interaction history — just the item's metadata.

### 9.2 Collaborative Filtering

Recommends based on PATTERNS OF USER BEHAVIOR (ratings/interactions), not item content.
- **User-based CF**: finds users SIMILAR to you (based on rating patterns) and recommends what they liked. Needs a user-item interaction matrix.
- **Item-based CF**: finds items similar to ones you already liked (based on how OTHER USERS rated them together).
- **Problem**: can't recommend brand-new items with zero ratings (the "cold start problem") — this is where content-based filtering is needed instead.

**Required data for user-based CF**: a user-item interaction matrix (ratings/purchases), NOT item content features or image embeddings (those are for content-based approaches).

### 9.3 Matrix Factorization / SVD

Breaks the big user-item matrix into smaller matrices representing hidden ("latent") factors — reduces dimensionality and can fill in predicted ratings for missing entries.

---

## 10. Time Series Analysis

### 10.1 Stationarity & ADF Test

A time series is "stationary" if its statistical properties (mean, variance) don't change over time — no trend, no changing spread.

**adfuller() function** (Augmented Dickey-Fuller test): tests whether a series is STATIONARY.
- Returns a p-value.
- **Interpretation** (standard hypothesis testing logic):
  - If p-value < 0.05 (significance level) → REJECT the null hypothesis → series IS stationary.
  - If p-value ≥ 0.05 → FAIL to reject null → series is NON-stationary.
- Example: p-value = 0.03 < 0.05 → series IS stationary.

*(Note: this is opposite of typical "p<0.05 means significant effect" thinking applied directly — but for ADF specifically, the null hypothesis is "series has a unit root" i.e. is non-stationary, so a SMALL p-value means you reject non-stationarity, concluding it IS stationary.)*

### 10.2 ARIMA Model

**A**uto**R**egressive **I**ntegrated **M**oving **A**verage — a classic time series forecasting model with three parameters `order=(p,d,q)`:
- **p**: number of past values used (autoregressive lag terms).
- **d**: number of times the data is DIFFERENCED (subtracting each value from the previous one) to make it stationary. This REMOVES trends.
- **q**: size of the moving average window (uses past forecast errors).

**Correct usage sequence:**
```python
model = ARIMA(data, order=(1,1,1))
model_fit = model.fit()          # returns a NEW fitted results object
print(model_fit.summary())        # call summary on the FITTED result, not the raw model
```
Common trap: calling `.summary()` on `model` BEFORE `.fit()`, or calling `.fit()` on the wrong object — always fit first, store the result, then use the result object for `.summary()` / `.forecast()` / `.predict()`.

**Handling NaNs**: statsmodels does NOT auto-impute — `.fit()` will error out or behave unpredictably if there are NaNs; you must clean data first.

---

## 11. Image Processing (PIL / Basic Computer Vision)

- Images loaded as NumPy arrays have shape `(height, width, channels)`. `channels=3` means RGB (3 color channels).
- Pixel intensity range for `uint8` images is **0 to 255** (256 possible values), NOT "up to 256" as a maximum value — 256 itself is never a valid pixel value (0-255 inclusive, so max is 255).
- `Image.resize((w,h))`: changes width/height — does NOT change the number of color channels.
- `.convert('L')`: converts an image to GRAYSCALE (single channel, Luminance) — this REDUCES channels from 3 to 1, it doesn't "flip" the image.
- `.flatten()`: converts a 2D/3D array into a single 1D array (unrolls all pixel values into one long list).

### Data Augmentation

Artificial modifications applied to training images to create more variety and reduce overfitting:
- Random cropping ✓ (augmentation)
- Rotation with small angle range ✓ (augmentation)
- Resizing to fixed resolution ✗ (this is just standard PREPROCESSING, needed for the model to accept consistent input sizes, not "augmentation" which implies creating variety)
- Normalizing pixel values to [0,1] ✗ (this is preprocessing/scaling, not augmentation)

---

## 12. Pandas / Data Manipulation

### 12.1 Boolean Filtering

`df[condition]` returns rows where `condition` is True.
Combine conditions with `&` (AND) or `|` (OR) — each condition MUST be wrapped in parentheses:
```python
df[(df['Age']>30) & (df['Salary']>70000) | ((df['Dept']=='IT') & (df['Experience']<6))]
```
Work through row by row, checking each condition, to see which rows survive.

### 12.2 groupby() variations

- **`.groupby(col).filter(func)`**: keeps or drops ENTIRE GROUPS based on whether `func` (applied to the whole group) returns True. Returns rows, same columns as original.
- **`.groupby(col).transform(func)`**: computes a per-group value and BROADCASTS it back to every row of that group (same length as original DataFrame) — useful for comparing a row to its group's mean.
- **`.groupby(col).apply(func)`**: most flexible — can return almost anything, but for filtering rows within groups by group AVERAGES, `.filter()` or `transform()` are the "textbook correct" tools depending on exactly what's needed.

**"Employees from departments where average salary > 65000"**: You want to keep ALL rows belonging to departments whose average salary exceeds 65000. Correct approaches:
- `df.groupby('Dept').filter(lambda x: x['Salary'].mean() > 65000)` — keeps whole groups matching the condition. ✓
- `df[df.groupby('Dept')['Salary'].transform('mean') > 65000]` — computes group mean per row, then filters. ✓ (also correct)

### 12.3 sort_values() and reset_index()

- `.sort_values('col')`: sorts rows by that column, ascending by default.
- `.reset_index(drop=True)`: resets the row index to 0,1,2,... AND drops the old index (if `drop=True`), instead of keeping it as a new column.

---

## 13. General Scikit-Learn API Concepts

### 13.1 sklearn's Design Philosophy

Core principles that guide how ALL sklearn estimators are built:
- **Consistency**: all estimators use `.fit()`, `.predict()`, `.transform()` with the SAME interface pattern.
- **Sensible defaults**: most parameters have good default values so beginners can start without heavy tuning.
- **Nonproliferation of classes**: datasets/parameters are represented with plain Python/NumPy structures rather than creating tons of custom special classes.
- **Direct accessibility of hyperparameters**: hyperparameters are stored as plain public attributes (like `model.alpha`) so you can inspect and set them directly.

(Trap: "execution of all code within 99.9ms" is NOT part of the design philosophy — it's made up.)

### 13.2 fit vs transform vs fit_transform

- **`.fit(X)`**: LEARNS parameters from the data (e.g., mean/std for a scaler, vocabulary for a vectorizer) — does not change the data.
- **`.transform(X)`**: APPLIES the already-learned parameters to transform data — used on data you're NOT learning from (e.g., test data).
- **`.fit_transform(X)`**: does BOTH at once — learn AND apply — used on TRAINING data.

**Golden rule**: fit (or fit_transform) ONLY on training data. Use `.transform()` (not `fit_transform`) on test/unseen data, so you don't "leak" test data statistics into your preprocessing.

- Fitting MinMaxScaler on TRAIN data → `fit_transform` ✓ correct.
- Fitting MinMaxScaler on TEST data → should be `.transform()` only (using train's fitted parameters), NOT `fit_transform` (that would leak test data statistics) — a common trap "fitting MinMaxScaler on test data - fit_transform" is WRONG/should be transform only.
- Training a decision tree: `.fit()` only (trees don't have a `.transform()` — they directly predict).
- Learning a logistic regression's weights: `.fit()`, not `fit_transform` (models don't "transform" data, they PREDICT).

### 13.3 Bunch objects vs DataFrames

- `sklearn.datasets.load_iris()`, `load_wine()`, `load_digits()`, `load_breast_cancer()`, etc. return a **Bunch object** (a dictionary-like structure with `.data`, `.target`, `.feature_names`, etc.) by default.
- `pd.read_csv(...)` returns a **pandas DataFrame**.
- `return_X_y=True` makes dataset loaders return `(X, y)` as arrays directly instead of a Bunch.
- `as_frame=True` makes the loader return DataFrames/Series instead of NumPy arrays.

### 13.4 Assertion-Reason Style Questions

These test whether you understand WHY something is true, not just whether it's true.
Example: "Assertion: precision should always be maximized over recall. Reason: recall is independent of false negatives."
- Assertion is FALSE (no universal priority — depends on the problem).
- Reason is also FALSE (recall = TP/(TP+FN), it's directly DEFINED using false negatives, not independent of them).
- So answer: "Both A and R are false."

**Strategy**: evaluate the Assertion and Reason SEPARATELY as true/false statements first, THEN check if the Reason actually logically explains the Assertion (only relevant if both are true).

---

## Quick Reference — Common Traps to Watch For

1. **"Always" / "Never" / "Guaranteed" statements** are usually FALSE in ML — almost nothing is guaranteed (e.g., "increasing n_estimators always improves accuracy" = FALSE).
2. **`.fit_transform()` on test data** = usually WRONG (data leakage) — should be `.transform()` only.
3. **K-Means/make_blobs cluster sizes** are NOT guaranteed equal, even with fixed n_samples and centers.
4. **Random Forest `max_features`** applies to EVERY tree, EVERY split — not just "some trees."
5. **ARIMA `.summary()`** must be called on the FITTED result (after `.fit()`), not the raw model object.
6. **R² score range** is -∞ to 1, NOT 0 to 1 (it CAN be negative).
7. **LabelEncoder vs MultiLabelBinarizer**: single label per row → LabelEncoder; multiple labels per row → MultiLabelBinarizer.
8. **liblinear solver** in LogisticRegression does NOT support true multinomial handling.
9. **Bagging reduces variance; Boosting reduces bias** — don't mix these up.
10. **StandardScaler/MinMaxScaler `.transform()` on new data** uses statistics learned during `.fit()` on OLD (training) data, not recalculated from the new data.

---

*Use this guide alongside the practice questions — for each question, first identify WHICH topic/concept it's testing (from the sections above), then apply the relevant rule or formula.*
