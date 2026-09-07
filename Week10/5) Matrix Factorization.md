## Topic 7: Matrix Factorization (SVD)

**The core idea:** So far, item-item and user-user CF worked by comparing rows/columns directly in the sparse matrix — and remember, that matrix is ~98% empty (Topic 2). Matrix factorization takes a different approach: instead of comparing raw rows, it tries to discover a small number of **hidden ("latent") factors** that explain the rating patterns, then uses those to fill in the blanks.

### The concept

```
Decompose R (users × movies) into:
R ≈ U · Σ · Vᵀ

U: (users × k)   : user latent factors
Σ: (k × k)       : singular values (importance of each factor)
V: (movies × k)  : item latent factors

k << min(n_users, n_movies): compressed representation
```

Think of "latent factors" as hidden axes of taste that the algorithm invents automatically — it doesn't label them, but you can imagine they might loosely correspond to things like "how much action vs. drama," "mainstream vs. arthouse," "how recent," etc. Nobody tells the algorithm what these axes mean; it finds whatever combination of hidden dimensions best reconstructs the original ratings.

- **U** (users × k): for each user, how much they lean toward each latent factor
- **Σ** (k × k): a diagonal matrix ranking how *important* each latent factor is to explaining the data overall
- **V** (movies × k): for each movie, how much it expresses each latent factor
- **k** is a number you choose, much smaller than the number of users or movies — this is the "compression." Multiplying U · Σ · Vᵀ back together gives you an approximation of the *full* ratings matrix — including the cells that were originally empty (0). That's the trick: it predicts values for movies a user never rated, based on the learned patterns.

**Why this helps with sparsity:** Instead of needing two movies to be co-rated by many of the same users (which item-item CF requires), SVD learns general taste dimensions from *all* the data at once, so it can make reasonable predictions even for sparse combinations.

### The code

```python
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=50, random_state=42)
U = svd.fit_transform(user_item)   # user latent factors (n_users, 50)
Sigma = np.diag(svd.singular_values_)
Vt = svd.components_               # (50, n_movies)

# Reconstruct predicted rating matrix
R_hat = U @ Sigma @ Vt             # (n_users, n_movies)
```

- `TruncatedSVD(n_components=50)` — this sets k=50 latent factors. (Regular SVD would give you as many factors as the smaller matrix dimension; "truncated" means we only keep the top 50 most important ones — a deliberate compression.)
- `svd.fit_transform(user_item)` — this does two things in one call: learns the decomposition *and* directly outputs `U` (the user-factor matrix).
- `svd.singular_values_` — the importance weights, turned into a diagonal matrix Σ via `np.diag`.
- `svd.components_` — this is `Vᵀ` directly (movies expressed as factors).
- `U @ Sigma @ Vt` — matrix multiplication reconstructs the *full* dense matrix, `R_hat`, of shape (users × movies) — now every cell has a predicted rating, even ones that were originally 0/empty.

`random_state=42` just makes the result reproducible (SVD's internal optimization has some randomness in initialization).

### Using it to recommend

```python
def svd_recommend(user_id, n=10):
    user_idx = user_item.index.get_loc(user_id)
    predicted_ratings = R_hat[user_idx]

    # Only recommend movies not yet rated
    rated_mask = user_item.iloc[user_idx].values == 0
    predicted_ratings[~rated_mask] = -1   # mask rated movies

    top_indices = predicted_ratings.argsort()[-n:][::-1]
    top_movies  = user_item.columns[top_indices]
    return movies[movies['movieId'].isin(top_movies)][['title','genres']]
```

Walking through it:
1. `user_item.index.get_loc(user_id)` — convert the actual `userId` into its positional row index (0, 1, 2...) since `R_hat` is a plain NumPy array, not indexed by ID.
2. `predicted_ratings = R_hat[user_idx]` — grab this user's full row of predicted ratings across *every* movie (including ones they already rated).
3. `rated_mask = user_item.iloc[user_idx].values == 0` — a boolean array: `True` where the *original* rating was 0 (unrated).
4. `predicted_ratings[~rated_mask] = -1` — the `~` flips the mask, so this targets movies the user **did** rate, and sets their predicted score to -1. This is a deliberate trick to push already-seen movies to the bottom, since you never want to "recommend" something they've already watched.
5. `predicted_ratings.argsort()[-n:][::-1]` — `argsort()` sorts ascending and returns *indices*; `[-n:]` grabs the top n indices (highest predicted scores); `[::-1]` reverses so the best is first.
6. Map those positional indices back to actual `movieId`s via `user_item.columns`, then look up titles.

**One subtlety worth flagging:** `R_hat` can technically produce values outside the 0.5–5.0 rating range (unlike the CF functions earlier, this one doesn't `np.clip`) — because SVD reconstruction is an unconstrained approximation, not a weighted average bounded by real rating values. That's fine for *ranking* purposes (we only care about relative order here), but if you wanted actual predicted star ratings from SVD, you'd want to clip it too.

---

Next: **Hybrid Recommenders** (Topic 8) — combining content + CF scores into one system, which is what real production recommenders actually do. Say "next" when ready.
