
## Topic 1: Neural Networks — Mathematical Foundations

### The Structure

```
Input layer → [Hidden layer 1] → [Hidden layer 2] → ... → Output layer
```

Each layer does **two** operations in sequence:

```
z = Wx + b     ← linear transform
a = f(z)       ← activation function
```

- `W` = weight matrix, `x` = input vector, `b` = bias vector
- `z` is a "raw" weighted sum (like linear regression)
- `f(z)` squashes/reshapes that sum non-linearly — this non-linearity is *why* neural nets can learn complex patterns. Stack only linear transforms and no matter how many layers you have, it collapses into one big linear function.

Two passes happen during training:
- **Forward pass**: push input through the layers to get a prediction
- **Backward pass**: compute how wrong you were, and push the error backward through the network to update every weight (this is **backpropagation**)

### Activation Functions

| Activation | Formula | Range | Where it's used |
|---|---|---|---|
| Sigmoid | 1/(1+e⁻ˣ) | (0,1) | Binary output layer |
| Tanh | (eˣ−e⁻ˣ)/(eˣ+e⁻ˣ) | (-1,1) | Hidden layers (older style) |
| ReLU | max(0,x) | [0,∞) | Hidden layers — **default choice today** |
| Leaky ReLU | max(0.01x, x) | (-∞,∞) | Fixes ReLU's "dead neuron" problem |
| ELU | x if x>0, else α(eˣ−1) | (-α,∞) | Smooth negative-side alternative |
| Softmax | eˣⁱ/Σeˣʲ | (0,1), sums to 1 | Multiclass output layer |

**Intuition**: ReLU is popular because it's cheap to compute and doesn't saturate for positive inputs (sigmoid/tanh flatten out at extremes, killing the gradient). Softmax is special — it's only used on the *output* layer for multiclass problems because it turns raw scores into a proper probability distribution.

### Loss Functions

| Task | Loss | Formula |
|---|---|---|
| Binary classification | Binary cross-entropy | −(y·log(ŷ) + (1−y)·log(1−ŷ)) |
| Multiclass | Categorical cross-entropy | −Σyₖ·log(ŷₖ) |
| Regression | MSE | Σ(y−ŷ)²/n |
| Regression (robust to outliers) | MAE | Σ\|y−ŷ\|/n |

The loss measures how wrong the network's prediction was. Training = trying to make this number as small as possible.

### Backpropagation (the chain rule, in plain terms)

```
∂L/∂W = ∂L/∂a · ∂a/∂z · ∂z/∂W

Update: W ← W − α · ∂L/∂W
```

This says: to know how much a weight `W` contributed to the final error `L`, you multiply together three "sensitivities":
1. How the loss changes with the layer's output (`∂L/∂a`)
2. How the output changes with the pre-activation `z` (`∂a/∂z`)
3. How `z` changes with the weight itself (`∂z/∂W`)

Then you nudge the weight a small step (`α` = learning rate) in the direction that *reduces* the loss.

### Worked Example — Forward Pass by Hand

Given weights and biases for one layer:
```
W1 = [[0.5, 0.5], [-1.0, 1.0]]
b1 = [0, 0.5]
input x = [1, -1]
```

**Step 1 — Dot product (z = Wx + b):**
- Neuron 1: `(1 × 0.5) + (-1 × 0.5) + 0 = 0`
- Neuron 2: `(1 × -1.0) + (-1 × 1.0) + 0.5 = -1.5`

**Step 2 — Apply ReLU (max(0, z)):**
- Neuron 1: `max(0, 0) = 0`
- Neuron 2: `max(0, -1.5) = 0`

**Result:** both neurons output 0 — this is called a **"dead layer"** for this particular input. It's a real practical danger with ReLU: if a neuron's inputs consistently push `z` negative, that neuron outputs 0 forever and its gradient is 0 too, so it stops learning entirely. (This is exactly why Leaky ReLU/ELU exist — they let a small negative signal through so the neuron can recover.)

---

That's the full math foundation section. Want me to continue to **Topic 2: MLPClassifier/MLPRegressor in sklearn** (the actual code you'd write), or do you want to sit with this one a bit longer — e.g. I could work through another forward-pass example with you, or diagram the network structure?
