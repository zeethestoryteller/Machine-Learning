### SGDClassifier (Stochastic Gradient Descent):
Scales to 10M+ samples; ideal for online and incremental learning with support for various loss functions like hinge, log_loss, and modified_huber.
### PassiveAggressiveClassifier:
Scales to 10M+ samples; well-suited for streaming text data and fast online classification updates where margin management is critical.
### LinearSVC:
Scales to 1M+ samples; an efficient linear support vector machine for large datasets, though it does not natively output calibrated probabilities (`predict_proba`).
### LogisticRegression (with `solver='saga'`):
Scales to 1M+ samples; optimal when dealing with large-scale sparse features and L1/elastic-net regularization.
### HashingVectorizer + SGD:
Scales to virtually infinite sample sizes ($\infty$); indispensable for high-dimensional NLP streams where the vocabulary size is unbounded or too large to fit in memory.

**Selection Guide:**

* Use **SGDClassifier** or **LogisticRegression(solver='saga')** when you need probabilistic outputs or flexible regularization at scale.
* Use **PassiveAggressiveClassifier** or **HashingVectorizer + SGD** when processing streaming text or infinite data streams.
* Use **LinearSVC** when training a fast linear margin classifier and probability estimates are not required.
