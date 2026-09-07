# Topic 8: Hybrid Recommenders

**The core idea:** Every method so far has a weakness — content-based ignores rating quality signals, CF fails on cold start, SVD is a black box. A **hybrid recommender** combines multiple scores into one, so the strengths of one method cover the weaknesses of another.

### The code

```python
def hybrid_recommend(user_id, movie_id, alpha=0.5):
    # Content score (0-1)
    content_score = content_similarity_score(user_id, movie_id)

    # CF score (scaled to 0-1)
    cf_raw = predict_rating_item_item(user_id, movie_id)
    cf_score = (cf_raw - 0.5) / 4.5

    return alpha * content_score + (1 - alpha) * cf_score
```

Walking through it:
- `content_score` — assumed to come from a helper (not fully shown in the doc, but implied: something measuring how well this movie matches the user's historical genre preferences, scaled 0–1).
- `cf_raw = predict_rating_item_item(...)` — reuses the item-item CF function from Topic 4, which returns a rating on the 0.5–5.0 scale.
- `cf_score = (cf_raw - 0.5) / 4.5` — this is a **min-max rescale**: it maps the 0.5–5.0 range down to 0–1, so it's on the *same scale* as `content_score`. This step matters a lot — you can't blend a 0–1 score with a 0.5–5.0 score directly, or one would dominate the sum purely due to scale, not actual importance.
- `return alpha * content_score + (1 - alpha) * cf_score` — a **weighted blend**. `alpha=0.5` means 50/50 trust between content and collaborative signals. If you set `alpha=0.8`, you're leaning heavily on content (useful for new users/items where CF has little data); `alpha=0.2` leans mostly on collaborative signal (useful once you have plenty of rating history).

**The production-scale version**, mentioned but not implemented in the doc:

```
score = w1*CF + w2*content + w3*popularity + w4*freshness + w5*diversity
```

This is the realistic picture: real systems blend in *more* than just two signals —
- **popularity** — trending/widely-liked items, useful as a fallback when personalization signal is weak
- **freshness** — newer items get a boost so the catalog doesn't stagnate on old favorites
- **diversity** — deliberately injecting variety so recommendations aren't all near-duplicates of each other (this is why Netflix doesn't show you 10 nearly identical thrillers just because you liked one)

Each `w` is a tunable weight, usually learned or A/B tested in production rather than hand-set like `alpha=0.5` here.

---

Next: **Evaluation Metrics** (Topic 9) — how you actually measure whether any of this is working. Say "next" when ready.
