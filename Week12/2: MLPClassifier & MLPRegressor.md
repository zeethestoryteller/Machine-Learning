## Topic 2: MLPClassifier / MLPRegressor (sklearn)

### Why Neural Networks MUST Be Scaled

This is the single most important practical rule in this section, so let's understand *why* before touching code.

**The problem:** Suppose you feed an unscaled feature like `salary = 100,000` directly into a neural network.

**The reasoning, step by step:**
1. In the first layer: `z = w·x + b`
2. If `x = 100,000`, then `z` becomes huge — even with small weights.
3. If you're using sigmoid or tanh activation, a huge `z` pushes `f(z)` instantly to its extreme (1 or -1) — this is called **saturation**.
4. At that saturated point, the *slope* of the activation function is essentially flat — the gradient is ≈0.
5. Since backprop updates weights using that gradient, **a zero gradient means the weight stops updating.** The network stops learning completely. This is the classic **vanishing gradient** problem.

**The fix:** Always scale your inputs (mean 0, std 1) with `StandardScaler` before feeding them into an MLP. This keeps `z` in a reasonable range so activations don't saturate.

### The Code

```python
from sklearn.neural_network import MLPClassifier, MLPRegressor

mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),  # 3 hidden layers
    activation='relu',                   # 'logistic','tanh','relu','identity'
    solver='adam',                       # 'sgd','adam','lbfgs'
    alpha=0.0001,                        # L2 regularisation
    batch_size=32,                       # mini-batch size
    learning_rate='adaptive',            # 'constant','invscaling','adaptive'
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,                 # use 10% of train as validation
    validation_fraction=0.1,
    n_iter_no_change=10,                 # patience
    random_state=42
)

# IMPORTANT: Always scale inputs
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

mlp_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp',    mlp)
])
mlp_pipe.fit(X_train, y_train)
print(f"Accuracy: {mlp_pipe.score(X_test, y_test):.4f}")

# Loss curve
import matplotlib.pyplot as plt
plt.plot(mlp.loss_curve_, label='Training loss')
if mlp.validation_scores_:
    plt.plot(mlp.validation_scores_, label='Validation score')
plt.xlabel('Iteration'); plt.legend()
```

**Walking through what each piece does:**
- `Pipeline` bundles the scaler and the model together so scaling always happens automatically and consistently — including at prediction time. This also prevents data leakage (a theme from earlier weeks in your course).
- `mlp.loss_curve_` is a list of the loss value at every iteration — plotting it tells you whether the model is still improving or has plateaued.
- `early_stopping=True` carves out 10% of the training data as an internal validation set and stops training once the validation score stops improving for `n_iter_no_change` rounds — this prevents overfitting and wasted compute.

### Hyperparameter Guide

| Parameter | Typical values | Effect |
|---|---|---|
| `hidden_layer_sizes` | (100,), (256,128), (128,64,32) | More units/layers = more capacity (but more overfitting risk) |
| `activation` | relu (default) | relu usually best for hidden layers |
| `solver` | adam (default) | lbfgs for small data; sgd for large data |
| `alpha` | 0.0001–0.01 | L2 regularisation strength; increase if overfitting |
| `batch_size` | 32–256 | Smaller batches = noisier updates, which can actually help generalization |
| `learning_rate_init` | 0.001–0.01 | Tune this when using adam |
| `max_iter` | 200–1000 | Increase if the loss curve is still decreasing at the end |
| `early_stopping` | True | Almost always turn this on in practice |

**Mental model for `hidden_layer_sizes=(256, 128, 64)`:** this builds 3 hidden layers with 256, then 128, then 64 neurons — a "funnel" shape that's common in practice, gradually compressing the representation before the output layer.

---

That's the sklearn implementation. Ready for **Topic 3: Transfer Learning**, or do you want to first try running the MLP code on a toy dataset to see the loss curve yourself?
