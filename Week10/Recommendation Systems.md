# Topic 1: Overview & The Cold Start Problem

**What this week is about:** You're building systems that predict what a user will like, using the MovieLens dataset (a classic dataset of user movie ratings). There are three core approaches:

1. **Content-based filtering** — recommends items similar to those a user already liked or interacted with, relying exclusively on the descriptive attributes and features of the items rather than user-to-user rating overlaps.
2. **Collaborative filtering** — recommend items that *similar users* liked, or items *similar in usage pattern* to ones a user liked
3. **Matrix factorization** — compress the whole user-item ratings table into hidden ("latent") patterns

**The pipeline** (from the architecture diagram):
```
User history (ratings/clicks) 
    → User-item matrix (sparse table: rows=users, cols=items) 
    → Model (CF / Content / MF) 
    → Ranked top-N recommendations
```

**The Cold Start Problem** — this is the single most important concept to understand before anything else:

- **New user problem:** A brand-new user has rated nothing. Collaborative filtering needs *overlapping ratings* to find similar users or items — with zero ratings, there's nothing to compare. CF fails completely.
- **New item problem:** A brand-new movie has zero ratings. No one has "co-rated" it with anything else, so CF can't place it relative to other movies.
- **Why content-based helps (partially):** Content-based filtering doesn't need ratings — it just needs the item's own features (genre, description). So a new *movie* can still be recommended based on its genre, even with zero ratings. But a new *user* still has no profile to base content recommendations on either way — neither method cleanly solves that half.

**Why this matters practically:** Real systems (Netflix, Spotify) use *hybrid* approaches specifically because of this — lean on content/popularity for new users and items, then switch to collaborative signals once enough interaction data builds up.

---
# Topic 2: The MovieLens Dataset (loading & inspecting)

MovieLens is a benchmark dataset for recommender systems: real users rated real movies. The "100K" version has ~100,000 ratings. Two files matter:

- `ratings.csv` → columns: `userId, movieId, rating, timestamp`
- `movies.csv` → columns: `movieId, title, genres`

Let's go through the code line by line.

```python
import pandas as pd, numpy as np

ratings = pd.read_csv('ratings.csv')   # userId, movieId, rating, timestamp
movies  = pd.read_csv('movies.csv')    # movieId, title, genres
merged  = ratings.merge(movies, on='movieId')
```

- `pd.read_csv` loads each file into a DataFrame (a table).
- `ratings.merge(movies, on='movieId')` joins the two tables on the shared `movieId` column — so now each rating row also has the movie's title and genres attached. This is a standard SQL-style inner join.

```python
print(ratings.shape)        # (100836, 4)
print(f"Users: {ratings['userId'].nunique()}")
print(f"Movies: {ratings['movieId'].nunique()}")
print(f"Rating range: {ratings['rating'].min()} - {ratings['rating'].max()}")
print(f"Avg rating: {ratings['rating'].mean():.2f}")
```

- `.shape` gives (rows, columns) — 100,836 ratings, 4 columns.
- `.nunique()` counts distinct values — how many unique users and movies exist.
- `.min()/.max()/.mean()` give you a quick statistical sense of the ratings (usually 0.5–5.0 stars).

**Why check this first?** Before building any model, you always want a sanity check: how big is the data, how sparse is it, what's the scale of ratings. This shapes every decision after.

### Sparsity — the most important number here

```python
n_users  = ratings['userId'].nunique()
n_movies = ratings['movieId'].nunique()
sparsity = ratings.shape[0] / (n_users * n_movies)
print(f"Sparsity: {sparsity*100:.2f}%")   # typically < 2%
```

Think of it this way: if you built a giant table of every user × every movie, `n_users * n_movies` is the total number of *possible* ratings (every cell). But users have only rated a tiny fraction of movies — `ratings.shape[0]` is the *actual* number of ratings that exist.

So `sparsity = actual ratings / possible ratings`. Typically this is under 2% — meaning 98%+ of the user-item matrix is empty. This single number is *why* collaborative filtering is hard, why cold start is a real problem, and why matrix factorization (compressing to latent factors) is useful later — you can't just brute-force compare full rows when almost everything is blank.

---
# Topic 3: The User-Item Matrix

This is the foundational data structure for every collaborative filtering method. Right now, `ratings` is a *long* table — one row per (user, movie, rating). We need to reshape it into a *wide* table: rows = users, columns = movies, cell values = ratings.

```python
user_item = ratings.pivot_table(
    index='userId', columns='movieId', values='rating', fill_value=0
)
print(user_item.shape)   # (users, movies)
```

**What `pivot_table` does:**
- `index='userId'` → each unique user becomes a row
- `columns='movieId'` → each unique movie becomes a column
- `values='rating'` → the cell at (user, movie) is filled with that user's rating for that movie
- `fill_value=0` → if a user never rated that movie, put `0` instead of a missing value (`NaN`)

Picture it like a spreadsheet:

| userId \ movieId | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| 1 | 5 | 0 | 3 | 0 |
| 2 | 0 | 4 | 0 | 0 |
| 3 | 2 | 0 | 0 | 5 |

Most of this table is `0` — that's the sparsity from Topic 2 made visible. A `0` here doesn't mean "rated zero stars," it means "hasn't rated this movie at all" — an important distinction to keep in your head, because later some formulas treat `0` as "unrated" and filter it out explicitly (you'll see `ratings[ratings > 0]` patterns for exactly this reason).

### Why sparse matrix representation matters

```python
from scipy.sparse import csr_matrix
user_item_sparse = csr_matrix(user_item.values)
```

If you have, say, 600 users and 9,000 movies, that's 5.4 million cells — but only ~100,000 are actually filled. Storing all 5.4 million values (mostly zeros) in memory is wasteful. `csr_matrix` (Compressed Sparse Row) is a data structure that only stores the *non-zero* values and their positions, saving huge amounts of memory on real-world datasets (which are far bigger than MovieLens 100K).

For learning/small datasets you can often get away with the dense `user_item` DataFrame directly (easier to read, index by name). But it's worth knowing `csr_matrix` exists for when data scales up.
