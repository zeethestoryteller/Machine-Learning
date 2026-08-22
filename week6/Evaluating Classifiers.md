Here are concise, structured notes on **Evaluating Classifiers** based on the material from your course material.

---

## 1. Stratified Cross-Validation

Standard cross-validation can suffer when class distributions are imbalanced across folds. Stratified splitting replicates the overall class distribution in every individual fold.

* **StratifiedKFold**: Divides data into $k$ folds while preserving the percentage of samples for each class.
* **RepeatedStratifiedKFold**: Repeats Stratified K-Fold multiple times with different randomization.
* **StratifiedShuffleSplit**: Generates randomized train/test split folds that preserve class proportions (folds may overlap).

---

## 2. LogisticRegressionCV & Hyperparameter Tuning

* Performs built-in cross-validation to optimize hyperparameters like **$C$** (inverse regularization strength) and **`l1_ratio`**.
* **`refit` parameter behavior**:
* `refit = True`: Averages scores across folds, selects hyperparameters with the best score, and refits the model on the full dataset using those parameters.
* `refit = False`: Averages the coefficients, intercepts, and $C$ values corresponding to the best scores across folds directly.



---

## 3. Classification Metrics (`sklearn.metrics`)

Common metrics used to evaluate classifier performance:

* `accuracy_score`: Fraction of correctly predicted samples.
* `balanced_accuracy_score`: Average accuracy obtained on each individual class (useful for imbalanced data).
* `top_k_accuracy_score`: Considers a prediction correct if the true label is among the top $k$ predicted classes.
* `precision_score`: Ratio of true positives to total predicted positives.
* `recall_score`: Ratio of true positives to total actual positives (True Positive Rate).
* `f1_score`: Harmonic mean of precision and recall.
* `roc_auc_score`: Area Under the Receiver Operating Characteristic curve.

---

## 4. Confusion Matrix

A tabular layout that visualizes the performance of a classification model.

* **Structure**: Row $i$ represents the true class, and column $j$ represents the predicted class. Entry $(i, j)$ is the number of observations actually in group $i$ but predicted as group $j$.
* **Display APIs**:
* `confusion_matrix(y_true, y_predicted)`
* `ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)`
* `ConfusionMatrixDisplay.from_predictions(y_test, y_pred)`



---

## 5. Classification Report

The `classification_report` function generates a text summary showing key metrics per class:

```python
from sklearn.metrics import classification_report
print(classification_report(y_true, y_predicted))

```

* **Metrics included**: Precision, recall, f1-score, and support (number of occurrences of each true class).
* **Averages included**: Macro average, weighted average, and overall accuracy.

---

## 6. Threshold-Based Metrics & Curves

* **Precision-Recall Curve**: Evaluates trade-offs across probability thresholds (`precision_recall_curve`).
* **ROC Curve**: Plots True Positive Rate (TPR) against False Positive Rate (FPR) at various classification thresholds (`roc_curve`).

---

## 7. Extending Binary Metrics to Multi-Learning

To apply binary evaluation metrics to multi-class or multi-label problems, binary metrics are computed per class and then aggregated using the **`average`** parameter:

* **`macro`**: Calculates the unweighted mean of the binary metrics for each class (treats all classes equally).
* **`weighted`**: Computes the average weighted by each class's support (its frequency in the true data).
* **`micro`**: Calculates metrics globally by counting the total true positives, false negatives, and false positives across all sample-class pairs.
* **`samples`**: Calculates metrics for each sample independently and returns their average (primarily used for multi-label tasks).
* **`None`**: Returns an array containing the score for each individual class.
