# Topic 4: ACF & PACF Interpretation

### Why this exists
Once your series is stationary, the next question is: **how much of `p` (AR order) and `q` (MA order) do I need for ARIMA?** ACF and PACF plots are the diagnostic tool that helps you read this off visually, before you even fit a model.

### What each plot actually measures

- **ACF (Autocorrelation Function)**: correlation between `y_t` and `y_{t-k}`, for each lag `k`. Importantly, this is the *total* correlation — it includes indirect effects passed through intermediate lags (e.g., the correlation between `y_t` and `y_{t-2}` partly flows through `y_{t-1}`).

- **PACF (Partial Autocorrelation Function)**: correlation between `y_t` and `y_{t-k}` **after removing the effect of all the lags in between** (1 through k-1). This isolates the *direct* relationship at exactly lag k.

This distinction — "total" vs "direct" correlation — is the whole key to why they're used together to distinguish AR from MA processes.

### The code

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(ts['diff1'].dropna(),  lags=40, ax=axes[0])
plot_pacf(ts['diff1'].dropna(), lags=40, ax=axes[1])
plt.tight_layout()
```

Note: you plot ACF/PACF on the **differenced (stationary)** series, not the raw one — ACF/PACF on non-stationary data just shows slow, misleading decay dominated by the trend, not the actual short-term dependency structure.

### Reading the plots — the core lookup table

| Pattern | ACF | PACF | Model |
|---|---|---|---|
| Significant at lag q, then cuts off | ✓ (cuts off after q) | Tails off | **MA(q)** |
| Tails off | Cuts off at lag p | ✓ (cuts off after p) | **AR(p)** |
| Both tail off | Tails off | Tails off | **ARMA(p,q)** |
| Slow decay | — | — | Non-stationary (difference first!) |
| Spike at lag s | — | — | Seasonal AR component |

### Why this pattern makes sense (not just a memorized table)

**AR(p) process**: each value directly depends on the last `p` values. So:
- PACF **cuts off sharply** after lag `p` — because once you've accounted for the direct effect of the last `p` lags, there's genuinely nothing left at lag `p+1`.
- ACF **tails off gradually** — because even though `y_t` only *directly* depends on `p` lags, those effects propagate forward indefinitely through the chain (like ripples), so correlation decays slowly rather than stopping abruptly.

**MA(q) process**: each value depends on the last `q` *shock/error* terms, not past values directly. So:
- ACF **cuts off sharply** after lag `q` — because the process has no memory beyond `q` steps by construction.
- PACF **tails off gradually** — removing indirect correlations still leaves a lingering, decaying signature since reconstructing the MA structure from lagged values needs an (in theory) infinite AR representation.

They're essentially mirror images of each other — which is a handy mnemonic: **"ACF cuts, it's MA. PACF cuts, it's AR."**

**"Slow decay" in both** almost always means you didn't fully remove non-stationarity — go back and difference more before trying to read `p`/`q` off these plots at all.

### Quick check
If your ACF plot shows a big spike at lag 1, then drops to near-zero and stays there for all further lags, but your PACF tails off slowly — what would you set `q` to in your ARIMA(p,d,q) model, roughly?

*(Answer: q=1 — the ACF cutting off sharply after lag 1 is the MA(1) signature.)*

Ready for **Topic 5: ARIMA & SARIMA**?
