# Cosine Similarity

This section covers how to measure how "similar" two documents are — the backbone of search engines and recommendation systems.

### The Intuition

```
Two documents can be long or short but discuss the same topic.
Cosine similarity measures the ANGLE between vectors: ignores magnitude.
```

**Why does this matter?** Imagine two documents about the same topic — one is a 3-sentence tweet, the other is a 2000-word article. Both mention "cats" and "veterinarian" proportionally the same amount, but the article's raw word-count vector will have much bigger numbers everywhere simply because it has more words. If you measured similarity using raw distance (like Euclidean distance), these two documents would look *far apart* just because of length — even though they're about the exact same thing.

Cosine similarity sidesteps this entirely by looking at the **direction** the vectors point in, not their length.

### The Formula

```
cos(θ) = (a·b) / (‖a‖ · ‖b‖)     ∈ [-1, 1]
For non-negative vectors (word counts): ∈ [0, 1]
1 = identical direction, 0 = orthogonal (no shared words)
```

Breaking this down:
- **`a·b`** (dot product) — multiply corresponding elements of the two vectors and sum them up. This is large when both vectors have big values in the same positions (i.e., they emphasize the same words).
- **`‖a‖` and `‖b‖`** (norms/magnitudes) — the "length" of each vector. Dividing by these normalizes out the effect of document length.
- The result is the cosine of the angle between the two vectors:
  - **1** = vectors point in exactly the same direction (perfectly similar word distribution)
  - **0** = vectors are orthogonal (perpendicular) — meaning they share *no* words in common
  - For text (where word counts/TF-IDF values are never negative), the range is practically **[0, 1]**, not the full [-1, 1] you'd get with vectors that can be negative.

### The Code

```python
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
import numpy as np

# Document similarity
X_tfidf = tfidf.fit_transform(docs)
sim_matrix = cosine_similarity(X_tfidf)    # (n_docs, n_docs)
```

This computes similarity between *every pair* of documents at once, producing a square matrix where entry `[i][j]` is the similarity between document i and document j. The diagonal will always be 1 (a document is perfectly similar to itself).

```python
# Find most similar document to query
query = tfidf.transform(["the cat sat on mat"])
sims = cosine_similarity(query, X_tfidf).flatten()
top_docs = sims.argsort()[::-1][:5]
```

This is the classic **search/retrieval pattern**:
1. Transform a new query string using the *already-fitted* vectorizer (note: `.transform()`, not `.fit_transform()` — you don't want to relearn the vocabulary from just the query).
2. Compute similarity of that single query vector against all documents in your corpus.
3. Sort descending (`[::-1]`) and take the top 5 — these are your "search results," ranked by relevance.

```python
# For sparse matrices, linear_kernel is faster (equivalent to cosine on L2-normalised vectors)
sims_fast = linear_kernel(query, X_tfidf).flatten()
```

**Why does this work?** `linear_kernel` just computes the raw dot product (no normalization step). But recall from Section 2 that `TfidfVectorizer` already applies `norm='l2'` by default — meaning every document vector is *already* unit length. If vectors are already unit length, the dot product **is** the cosine similarity (since dividing by `‖a‖·‖b‖` where both norms = 1 does nothing). So `linear_kernel` gives the identical result but skips the redundant normalization step, making it faster on large sparse matrices.

### Concept Breakdown: Why Cosine Similarity for Text?

**Problem:** Document A is a 500-word review. Document B is a 5-word summary of the same review.

**Reasoning:** Using Euclidean distance, these two documents would appear very far apart purely because Document A has much higher word counts than Document B — even if they're about the exact same thing.

**Result:** Cosine similarity looks only at the *angle*, ignoring magnitude. If the proportion of words used is similar between the two documents, the angle is small (similarity is high). This makes cosine similarity naturally **length-invariant** — a critical property for comparing documents of very different sizes.

---

That wraps up the NLP half of the document. Next up: **Section 6 — Image Processing for Classical ML** (turning images into feature vectors: pixel flattening, HOG features, and the standard image preprocessing pipeline). Continue?


-------
# Tiny concrete example by hand so you can see exactly what's happening.

### Example: Two documents

- **Doc A:** "cat sat mat"
- **Doc B:** "cat cat cat sat mat mat mat mat"

Notice Doc B is just Doc A but with way more repetition — same *topic*, wildly different *length*.

### Step 1: Build word count vectors

Vocabulary (alphabetical): `[cat, mat, sat]`

| Doc | cat | mat | sat |
|---|---|---|---|
| A | 1 | 1 | 1 |
| B | 3 | 4 | 1 |

So:
- **a** = [1, 1, 1]
- **b** = [3, 4, 1]

### Step 2: Compute Euclidean distance (to show why it's misleading)

```
distance = √[(1-3)² + (1-4)² + (1-1)²]
         = √[4 + 9 + 0]
         = √13
         ≈ 3.61
```

That's a fairly large distance — Euclidean distance says these documents look pretty different, purely because B has bigger numbers everywhere. But intuitively, both documents are 100% about "cat, mat, sat" in similar *proportions*. Euclidean distance is fooled by length.

### Step 3: Compute Cosine Similarity instead

**Dot product (a·b):**
```
(1×3) + (1×4) + (1×1) = 3 + 4 + 1 = 8
```

**Norms (magnitudes):**
```
‖a‖ = √(1² + 1² + 1²) = √3 ≈ 1.732
‖b‖ = √(3² + 4² + 1²) = √26 ≈ 5.099
```

**Cosine similarity:**
```
cos(θ) = 8 / (1.732 × 5.099)
       = 8 / 8.831
       ≈ 0.906
```

### The takeaway

Cosine similarity gives **≈0.91** — very high, correctly recognizing these two documents are about the same thing, just at different lengths/repetition levels. Euclidean distance completely missed this because it's sensitive to magnitude, not direction.

That's the exact "Document A is 500 words, Document B is a 5-word summary" scenario from the concept box in the PDF, just scaled down to numbers you can compute by hand.

Want me to visualize this as a diagram (the actual angle between the two vectors in 2D/3D), or move on to Section 6 (Image Processing)?
