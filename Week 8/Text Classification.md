```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, GridSearchCV

text_pipe = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', sublinear_tf=True)),
    ('clf',   LogisticRegression(max_iter=1000, class_weight='balanced'))
])

# Cross-validate (works because TF-IDF is inside pipeline)
scores = cross_val_score(text_pipe, X_train_text, y_train, cv=5, scoring='f1_macro')
print(f"F1: {scores.mean():.3f} ± {scores.std():.3f}")

# Tune
param_grid = {
    'tfidf__ngram_range': [(1,1),(1,2),(1,3)],
    'tfidf__max_features': [5000, 20000],
    'tfidf__sublinear_tf': [True, False],
    'clf__C': [0.1, 1, 10]
}
gs = GridSearchCV(text_pipe, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
gs.fit(X_train_text, y_train)
print(gs.best_params_)
```

| Classifier | Strength | Notes |
| --- | --- | --- |
| Logistic Regression | Strong baseline | Fast, interpretable |
| Linear SVC | Often best linear | No `predict_proba` natively |
| Multinomial NB | Very fast, sparse | Good for word counts |
| Complement NB | Imbalanced text | Improve on MultinomialNB |
| SGD (log_loss) | Very large corpora | Online learning |
