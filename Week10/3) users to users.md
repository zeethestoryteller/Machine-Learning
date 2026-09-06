# Topic 5: User-User Collaborative Filtering

**The core idea (flipped from item-item):** Instead of asking "which movies are similar," we ask "which *users* are similar to me?" If User B has similar taste to User A, then movies B loved (that A hasn't seen) get recommended to A.

### Step 1: Build the user-user similarity matrix

```python
user_similarity = cosine_similarity(user_item)
user_sim_df = pd.DataFrame(user_similarity,
                            index=user_item.index,
                            columns=user_item.index)
```

Notice: no `.T` this time. `user_item` is already (users × movies), and each **row is a user's rating vector** across all movies — which is exactly what we want to compare user-to-user. (Compare this to item-item CF, where we transposed to get movie rows.)

### Step 2: Predict a rating using similar users

```python
def predict_rating_user_user(user_id, movie_id, k=20):
    if movie_id not in user_item.columns:
        return user_item.loc[user_id][user_item.loc[user_id] > 0].mean()

    sim_scores = user_sim_df[user_id].drop(user_id).sort_values(ascending=False)

    # Only users who rated this movie
    raters = user_item[movie_id]
    raters = raters[raters > 0]
    sim_raters = sim_scores[sim_scores.index.isin(raters.index)]

    if len(sim_raters) == 0:
        return user_item[movie_id][user_item[movie_id]>0].mean()

    top_users = sim_raters.head(k)
    ratings = user_item.loc[top_users.index, movie_id]
    prediction = (top_users.values * ratings.values).sum() / top_users.sum()
    return np.clip(prediction, 0.5, 5.0)
```

Walking through it:
1. If the movie doesn't exist in our data at all → fall back to the user's own average rating.
2. `sim_scores = user_sim_df[user_id].drop(user_id)` — get this user's similarity to every other user, dropping self-similarity (always 1, useless).
3. `raters = user_item[movie_id]` then filter `> 0` — find every user who *actually rated this specific movie*. This step is necessary because most users haven't rated most movies (sparsity again).
4. `sim_raters = sim_scores[...isin(raters.index)]` — narrow the similarity list down to *only* users who rated the target movie. No point weighting by similarity to someone who can't contribute a rating.
5. If nobody similar rated this movie → fall back to the overall average rating for that movie across everyone.
6. `top_users = sim_raters.head(k)` — take the k=20 most similar qualifying users.
7. Same **weighted average** formula as item-item: similarity-weighted average of their ratings for this movie.
8. Clip to valid rating range.

### Pen & paper example (from the doc) — walk through it by hand

**Given:**
- User A: Movie1=5, Movie2=4, Movie3=**?** (want to predict)
- User B: Movie1=4, Movie2=4, Movie3=5
- User C: Movie1=1, Movie2=2, Movie3=2

**Step 1 — Similarity:** Compare A and B/C using only *co-rated* items (Movie1, Movie2 — since Movie3 is what we're predicting). A's pattern `[5,4]` is close to B's `[4,4]` (both high) and far from C's `[1,2]` (both low). So Sim(A,B) is high, Sim(A,C) is low.

**Step 2 — Isolate raters:** Who rated Movie3? Both B (5) and C (2).

**Step 3 — Weighted average:**
$$\text{Prediction} = \frac{\text{Sim}_{AB} \times \text{Rating}_{B3} + \text{Sim}_{AC} \times \text{Rating}_{C3}}{\text{Sim}_{AB} + \text{Sim}_{AC}}$$

Since Sim_AB dominates, the prediction lands close to B's rating (5) — the math naturally "listens more" to the taste-twin and mostly ignores the dissimilar user, even though both technically rated the movie.

### Item-Item vs User-User — when to use which

| Feature | Item-Item CF | User-User CF |
|---|---|---|
| Similarity between | Items | Users |
| Scales better when | Many users, stable items | Many items, stable users |
| More stable | ✓ (items change less) | ✗ (tastes evolve over time) |
| Industry usage | More common | Less common |

**Why item-item wins in practice:** A movie's "identity" (who tends to like it) barely changes month to month. A *user's* taste, on the other hand, can shift — new interests, mood, life stage. Recomputing user-user similarity constantly is expensive and the results go stale faster. This is exactly why Amazon's original recommender (and most production systems) use item-item, not user-user.

---

Next: **Content-Based Filtering** (Topic 6), which sidesteps ratings entirely and uses genres/text instead. Say "next" when ready.