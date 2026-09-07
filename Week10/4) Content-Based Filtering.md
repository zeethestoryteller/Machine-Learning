# Topic 6: Content-Based Filtering

**The core idea (completely different from CF):** Forget what other users did. Look at the *item itself* — its genres, description, tags — and recommend movies that are similar in content. This is why content-based filtering solves the *item* cold start problem from Topic 1: a brand-new movie has genres from day one, even with zero ratings.

### Step 1: Turn genres into numeric vectors

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies['genres_str'] = movies['genres'].str.replace('|', ' ')

tfidf = TfidfVectorizer(stop_words='english')
genre_matrix = tfidf.fit_transform(movies['genres_str'])
```

- In the raw data, genres look like `"Action|Adventure|Sci-Fi"` — pipe-separated. `.str.replace('|', ' ')` turns that into `"Action Adventure Sci-Fi"`, a space-separated string, so it looks like a little "document" of words.
- `TfidfVectorizer` converts each movie's genre string into a numeric vector. **TF-IDF** stands for Term Frequency–Inverse Document Frequency: it weights words by how often they appear *for this movie* vs. how common they are *across all movies*. A genre like "Sci-Fi" that appears in only a few movies gets a higher weight (more distinctive) than a genre like "Drama" that appears everywhere (less distinctive, since it doesn't help discriminate between movies).
- `stop_words='english'` strips generic filler words — not very relevant for genre tags, but standard practice for text vectorization.
- Result: `genre_matrix` is (movies × unique genre-words), where each row is a movie's genre "fingerprint" as numbers.

### Step 2: Compare movies by content similarity

```python
content_sim = cosine_similarity(genre_matrix, genre_matrix)
```

Same cosine similarity trick as before — but now it's comparing genre vectors instead of rating patterns. Two movies tagged `Action Sci-Fi` and `Action Sci-Fi Thriller` will have high similarity because they share genre words.

### Step 3: Recommend based on content

```python
def content_based_recommend(movie_id, n=10):
    idx = movies[movies['movieId'] == movie_id].index[0]
    sim_scores = list(enumerate(content_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices][['title','genres']]
```

Walking through it:
1. `idx = movies[movies['movieId'] == movie_id].index[0]` — find the row position of this movie in the `movies` DataFrame (not the movieId itself, but its positional index, since `content_sim` is a plain array indexed 0, 1, 2...).
2. `content_sim[idx]` — grab that movie's row of similarity scores against every other movie.
3. `list(enumerate(...))` — pairs each score with its position: `[(0, 0.2), (1, 0.9), (2, 0.05), ...]`.
4. `sorted(..., key=lambda x: x[1], reverse=True)` — sort by similarity score, descending.
5. `[1:n+1]` — **skip index 0**. The most similar movie to itself is always itself (similarity = 1), so this slice skips that and takes the next `n` most similar.
6. Map the resulting positions back to actual movie rows and return `title`/`genres`.

**Key contrast with CF:** notice this function needs *no rating data at all* — it works purely off the `movies` table. That's exactly why it works for brand-new movies with zero interactions, but it *also* means it can't capture nuance that ratings would reveal — e.g., two Action movies with identical genre tags could be wildly different quality, and content-based filtering has no way to know that. It only knows "these look similar on paper."

---

Next: **Matrix Factorization (SVD)** (Topic 7) — a fundamentally different technique that compresses the whole sparse matrix into hidden "taste dimensions." Say "next" when ready.
