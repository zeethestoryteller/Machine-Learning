# Topic 4: Item-Item Collaborative Filtering

**The core idea:** If users who like movie A also tend to like movie B, then A and B are "similar" — not because they share genres, but because they share a *rating pattern* across users. If you liked A and haven't seen B, B gets recommended.

This is why it's called *collaborative* — the similarity comes from the collective behavior of many users, not from the movie's actual content.

### Step 1: Build the item-item similarity matrix

```python
from sklearn.metrics.pairwise import cosine_similarity

item_similarity = cosine_similarity(user_item.T)
item_sim_df = pd.DataFrame(item_similarity,
                            index=user_item.columns,
                            columns=user_item.columns)
```

- `user_item` is (users × movies). We `.T` (transpose) it to (movies × users) — because now each **row is a movie**, represented as a vector of ratings it received from every user.
- `cosine_similarity` compares every pair of these movie-vectors and outputs a score between -1 and 1 (in practice, 0 to 1 here since ratings are non-negative). A score close to 1 means two movies were rated similarly by the same users.
- Wrapping it in a DataFrame with `index=columns=user_item.columns` lets you look up similarity by actual `movieId`, e.g. `item_sim_df[1][50]` = similarity between movie 1 and movie 50.

**What cosine similarity actually measures:** the angle between two vectors, ignoring magnitude. Two movies rated `[5,4,3]` and `[5,4,3]` by three users → perfectly similar (angle 0, similarity 1). This makes it robust to some users rating everything higher/lower overall — it cares about the *pattern* of ratings, not the absolute scale.

### Step 2: Find similar movies

```python
def get_similar_movies(movie_id, n=10):
    sims = item_sim_df[movie_id].drop(movie_id).sort_values(ascending=False)
    top = sims.head(n).reset_index()
    top.columns = ['movieId', 'similarity']
    return top.merge(movies[['movieId','title']], on='movieId')
```

- `item_sim_df[movie_id]` grabs the similarity column for one movie — a score against every other movie.
- `.drop(movie_id)` removes the movie's similarity with *itself* (which would always be 1, a useless result).
- `.sort_values(ascending=False)` puts the most similar movies first.
- `.head(n)` takes the top 10.
- The rest just merges in movie titles so you get human-readable names instead of raw IDs.

### Step 3: Predict a rating a user *would* give a movie

This is the more powerful use — not just "what's similar" but "what rating would user X give movie Y."

```python
def predict_rating_item_item(user_id, movie_id, k=20):
    user_ratings = user_item.loc[user_id]
    rated_movies = user_ratings[user_ratings > 0]

    if movie_id not in item_sim_df.columns:
        return user_ratings[user_ratings > 0].mean()

    sims = item_sim_df[movie_id][rated_movies.index]
    sims = sims[sims > 0]

    if len(sims) == 0:
        return rated_movies.mean()

    top_sims = sims.nlargest(k)
    prediction = (top_sims * rated_movies[top_sims.index]).sum() / top_sims.sum()
    return np.clip(prediction, 0.5, 5.0)
```

Walking through this step by step:
1. `user_ratings = user_item.loc[user_id]` — grab that user's full row (ratings for all movies, mostly 0s).
2. `rated_movies = user_ratings[user_ratings > 0]` — filter to just the movies they *actually* rated.
3. If the target movie isn't in our similarity matrix at all (edge case, e.g. brand new item) → just fall back to the user's average rating.
4. `sims = item_sim_df[movie_id][rated_movies.index]` — for the movie we're predicting, get its similarity to every movie the user *has* rated.
5. `sims[sims > 0]` — drop movies with zero/negative similarity (not useful signal).
6. If no similar rated movies exist → fall back to the user's average.
7. `top_sims = sims.nlargest(k)` — take the k=20 *most similar* rated movies (not all of them — this limits noise from weakly-related movies).
8. **The prediction formula** — this is a **weighted average**:
$$\text{prediction} = \frac{\sum (\text{similarity}_i \times \text{rating}_i)}{\sum \text{similarity}_i}$$
   Movies more similar to the target movie get more "voting power" in predicting the rating. If a user rated a very similar movie 5 stars, that pulls the prediction toward 5 much more than a weakly similar movie rated 5 stars.
9. `np.clip(prediction, 0.5, 5.0)` — ratings can only be 0.5 to 5.0, so this forces the result back into valid range even if the math produces something outside it.

**Intuition check:** this is exactly the same logic as "movies similar to A" — just applied to *predict a number* instead of *ranking a list*.
