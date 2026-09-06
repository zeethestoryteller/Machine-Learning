# 1. K-Means Clustering

**The idea:** You have unlabeled data and you want to group it into *k* clusters. K-Means does this by picking *k* "centroids" (cluster centers) and assigning every point to whichever centroid is closest.

**The algorithm (EM-like loop):**
1. **Initialize** — place k centroids, either randomly or using `k-means++` (a smarter method that spreads initial centroids apart so you don't get unlucky starts)
2. **E-step (Expectation/Assignment)** — for every point, find the nearest centroid and assign the point to that cluster
3. **M-step (Maximization/Update)** — recompute each centroid as the mean (average position) of all points assigned to it
4. **Repeat** steps 2-3 until centroids stop moving (convergence)

It's called "EM-like" because it alternates between assigning points (E) and updating parameters (M) — similar in spirit to the Expectation-Maximization algorithm used elsewhere in statistics.

**The objective it minimizes** — WCSS (Within-Cluster Sum of Squares), also called inertia:

$$WCSS = \sum_i \sum_{x \in C_i} \|x - \mu_i\|^2$$

In plain English: for every cluster, add up the squared distance from each point to its cluster's centroid, then sum across all clusters. Lower WCSS = tighter, more compact clusters.

**The code:**
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ALWAYS scale before clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```
**Why scale first?** K-Means uses Euclidean distance. If one feature is "income" (range: 0–200,000) and another is "age" (range: 0–100), income will dominate the distance calculation purely because of its scale — not because it's actually more important. Scaling puts every feature on equal footing (mean 0, std 1).

```python
kmeans = KMeans(
    n_clusters=5,
    init='k-means++',   # smart init (default, avoids bad random starts)
    n_init=10,           # run 10 times, pick best WCSS
    max_iter=300,
    tol=1e-4,
    random_state=42
)
labels = kmeans.fit_predict(X_scaled)
```
- `n_clusters=5` → you're telling it to find 5 groups (this is the "k" you need to choose — see below)
- `n_init=10` → K-Means can get stuck in a bad local optimum depending on where centroids start. Running it 10 times with different starts and keeping the best result (lowest WCSS) protects against this
- `tol=1e-4` → how small the centroid movement needs to be before we declare convergence

```python
print(f"Inertia (WCSS): {kmeans.inertia_:.1f}")
print(f"Cluster sizes: {pd.Series(labels).value_counts().sort_index().values}")
```
`kmeans.inertia_` gives you the final WCSS value — useful for comparing different values of k.

### Finding the optimal k

This is the catch with K-Means: **you have to tell it k in advance**. So how do you pick it?

```python
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

inertias, sil_scores, db_scores, ch_scores = [], [], [], []
K_range = range(2, 15)

for k in K_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    labels_k = km.labels_
    sil_scores.append(silhouette_score(X_scaled, labels_k))
    db_scores.append(davies_bouldin_score(X_scaled, labels_k))
    ch_scores.append(calinski_harabasz_score(X_scaled, labels_k))

best_k_sil = list(K_range)[np.argmax(sil_scores)]
```

You loop over candidate k values (2 through 14) and compute several metrics for each, then pick whichever k looks best:

| Metric | Range | Better when | Needs labels? |
|---|---|---|---|
| Silhouette Score | -1 to 1 | Higher | No (internal) |
| Davies-Bouldin | ≥0 | Lower | No (internal) |
| Calinski-Harabasz | ≥0 | Higher | No (internal) |
| Inertia (WCSS) | ≥0 | Lower | No (elbow method) |
| ARI | -1 to 1 | Higher | **Yes** |
| NMI | 0 to 1 | Higher | **Yes** |

**Silhouette score intuition:** for each point, it compares "how close am I to my own cluster" vs "how close am I to the nearest other cluster." Values near 1 = well-clustered, near 0 = on a boundary, negative = probably in the wrong cluster.

**The Elbow method:** plot inertia against k. Inertia always decreases as k increases (more clusters = tighter fit), but at some point the improvement flattens out — that "elbow" bend is your sweet spot. Beyond it, you're just overfitting with too many clusters.

ARI and NMI are only usable when you actually know the true labels (e.g., testing an algorithm on labeled data) — not useful for real unsupervised problems where you don't have ground truth.

---

## 2. Hierarchical Agglomerative Clustering (HAC)

**The idea:** Instead of picking k upfront, build a whole tree (hierarchy) of clusters from the bottom up, then decide afterward where to "cut" the tree to get k clusters.

**Algorithm:**
1. Start with every single point as its own cluster
2. Repeatedly merge the two *closest* clusters into one
3. Keep going until only one giant cluster remains
4. Cut the resulting tree (dendrogram) at whatever level gives you the number of clusters you want

**The key design choice: linkage** — this defines what "distance between two clusters" even means, since clusters (unlike points) contain multiple points:

| Linkage | How distance is measured | Cluster shape | Sensitive to outliers |
|---|---|---|---|
| single | minimum distance between any two points across clusters | chain-like, elongated | Very |
| complete | maximum distance between any two points | compact | Yes |
| average | average of all pairwise distances | balanced | Moderate |
| ward | minimizes increase in within-cluster variance after merging | compact, equal-sized | No — **usually the best default** |

**Code:**
```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Dendrogram (on a subsample for large data)
linkage_matrix = linkage(X_scaled[:200], method='ward')
plt.figure(figsize=(12,6))
dendrogram(linkage_matrix, truncate_mode='lastp', p=20,
           leaf_rotation=90, leaf_font_size=10)
plt.axhline(y=10, color='r', linestyle='--')  # cut level
plt.title('Dendrogram'); plt.show()
```
The dendrogram is a tree diagram — the height of each merge shows how "far apart" the two things being merged were. Drawing a horizontal line (`axhline`) across it and counting how many vertical lines it crosses tells you how many clusters you'd get if you cut there.

`X_scaled[:200]` — only the first 200 points are used for the dendrogram plot, because dendrograms get visually unreadable (and computationally expensive) with large datasets.

```python
agg = AgglomerativeClustering(
    n_clusters=5,
    linkage='ward',
    metric='euclidean'
)
labels = agg.fit_predict(X_scaled)
```
Here you still specify `n_clusters` — you've decided the cut point after inspecting the dendrogram.

**When to prefer HAC over K-Means:** small datasets where you want to actually see the hierarchy/structure. It doesn't scale well to large data (unlike K-Means).

---

## 3. DBSCAN (Density-Based Spatial Clustering)

**The idea:** Instead of clusters being defined by centroids, define them by *density* — regions where points are packed closely together are clusters; sparse/isolated points are noise.

**Two parameters:**
- **eps (ε)** — radius defining a point's "neighborhood"
- **min_samples** — how many neighbors a point needs (within eps) to count as a "core point"

**Point types:**
- **Core point:** has ≥ `min_samples` points within distance `eps`
- **Border point:** within `eps` of a core point, but doesn't itself have enough neighbors to be core
- **Noise point:** isolated — not reachable from any core point. Labeled `-1`

**Strengths:** finds clusters of any shape (not just round/convex blobs like K-Means), and automatically flags outliers as noise instead of forcing them into a cluster.
**Weaknesses:** struggles when clusters have very different densities, and `eps` can be tricky to choose.

**Code:**
```python
from sklearn.cluster import DBSCAN

db = DBSCAN(
    eps=0.5,
    min_samples=5,
    metric='euclidean',
    n_jobs=-1
)
labels = db.fit_predict(X_scaled)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = np.sum(labels == -1)
```
Note the `-1` label is reserved for noise, not an actual cluster — so it's subtracted out when counting clusters.

**Choosing eps — the k-distance plot trick:**
```python
from sklearn.neighbors import NearestNeighbors

nn = NearestNeighbors(n_neighbors=5)
nn.fit(X_scaled)
distances, _ = nn.kneighbors(X_scaled)
distances = np.sort(distances[:, -1])
plt.plot(distances)
plt.xlabel('Points sorted by distance'); plt.ylabel('5-NN distance')
# Look for the "elbow" → that is the eps value
```
For every point, find the distance to its 5th nearest neighbor, sort all these distances, and plot them. Where the curve suddenly bends upward (the "elbow") is roughly the density threshold separating "normal" regions from sparse ones — that's your `eps`.

**Algorithm comparison so far:**

| Algorithm | Shapes | Detects noise | Needs k upfront | Scales to big data |
|---|---|---|---|---|
| K-Means | Convex only | No | Yes | Great |
| HAC | Any | No | Yes | Poor |
| DBSCAN | Any | Yes | No (uses ε, min_samples) | OK |
| HDBSCAN | Any | Yes | No | Good |
| GMM | Elliptical | Soft/no | Yes | OK |

---

## 4. Gaussian Mixture Models (GMM)

**The idea:** Instead of hard-assigning each point to exactly one cluster, assume the data was generated by a mixture of several Gaussian (bell-curve) distributions, and figure out the parameters of each Gaussian plus the *probability* each point belongs to each one.

This gives you **soft clustering** — a point can be "70% cluster A, 30% cluster B" instead of a forced single label. It also handles elliptical (stretched) cluster shapes, unlike K-Means which assumes round/convex clusters.

**Code:**
```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(
    n_components=5,
    covariance_type='full',  # 'full','tied','diag','spherical'
    n_init=3,
    random_state=42
)
gmm.fit(X_scaled)
labels = gmm.predict(X_scaled)
probs  = gmm.predict_proba(X_scaled)  # soft assignments!
```
`covariance_type` controls the shape flexibility of each Gaussian "blob":
- `full` — each cluster can be any elliptical shape/orientation (most flexible, most parameters)
- `tied` — all clusters share the same shape
- `diag` — axes-aligned ellipses only
- `spherical` — perfectly round (most restrictive, like K-Means)

**Choosing n_components (like choosing k):** use BIC (Bayesian Information Criterion), which balances how well the model fits against how many parameters it uses (penalizing overly complex models):
```python
bics = [GaussianMixture(n_components=k, random_state=42).fit(X_scaled).bic(X_scaled)
        for k in range(1, 15)]
best_k = np.argmin(bics) + 1
```
Lower BIC is better here (unlike silhouette, where higher is better) — so we take the `argmin`.

---

## 5. Hyperparameter Tuning

Once you've picked a model (could be a classifier, regressor, or even one of the clustering models above with tunable parameters), you need to find the best settings for it. Three approaches, in increasing sophistication:

**Decision rule from the notes:**
- ≤3 hyperparameters, all discrete values → **GridSearchCV**
- More than 3, or continuous ranges → **RandomizedSearchCV**
- Large search space + expensive model to train → **Bayesian Optimization** (Optuna, Hyperopt)

### GridSearchCV — exhaustive search
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 5, 7, None],
    'min_samples_leaf': [1, 5, 10],
    'max_features': ['sqrt', 'log2', 0.5]
}
# Total combinations: 3×4×3×3 = 108 × 5 CV folds = 540 fits!

gs = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    refit=True,       # refit best model on full train set
    verbose=1,
    return_train_score=True
)
gs.fit(X_train, y_train)
print(gs.best_params_, gs.best_score_)
```
GridSearch literally tries **every combination** of the parameter values you give it. This is thorough but expensive — the comment shows how fast it explodes: 108 combos × 5-fold cross-validation = 540 total model fits, just for this one grid.

### RandomizedSearchCV — sample randomly instead
```python
from scipy.stats import uniform, randint, loguniform

param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(2, 20),
    'min_samples_leaf': randint(1, 50),
    'max_features': uniform(0.1, 0.9),
    'learning_rate': loguniform(0.001, 0.3)  # log-uniform for scale-invariant params
}

rs = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_dist,
    n_iter=100,     # number of random combinations to try
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42
)
rs.fit(X_train, y_train)
```
Instead of trying every combination, it randomly samples `n_iter` combinations from the specified distributions. This scales much better when you have many hyperparameters or continuous ranges (since a grid over continuous values would be infinite).

**Why `loguniform` for learning_rate?** Learning rate typically matters on a *multiplicative* scale — going from 0.001 to 0.01 is as meaningful a jump as going from 0.01 to 0.1. A regular uniform distribution would oversample large values and undersample the small ones that often matter most.

**⚠️ The n_iter vs cv trap (important distinction the notes call out):**
- `n_iter` = how many different parameter *combinations* to try
- `cv` = how many folds to train/validate *each* combination on
- **Total model fits = n_iter × cv**

So `n_iter=100, cv=5` → 500 actual model fits. Increasing `n_iter` explores more of the hyperparameter space; increasing `cv` gives a more statistically reliable score estimate for each point you've already sampled — they solve different problems.

### Successive Halving — faster alternative
```python
from sklearn.model_selection import HalvingRandomSearchCV

hs = HalvingRandomSearchCV(
    RandomForestClassifier(), param_dist,
    factor=3, cv=5, n_jobs=-1, random_state=42
)
hs.fit(X_train, y_train)
```
Starts by evaluating *many* candidates on a *small* amount of data/resources, then only keeps the best fraction (`1/factor`) and gives survivors more resources in the next round. Much faster than full grid/random search because bad candidates get eliminated early instead of running to completion.

### Bayesian Optimization with Optuna
```python
import optuna
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        'n_estimators':    trial.suggest_int('n_estimators', 50, 500),
        'learning_rate':   trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'max_depth':       trial.suggest_int('max_depth', 2, 8),
        'subsample':       trial.suggest_float('subsample', 0.5, 1.0),
        'min_samples_leaf':trial.suggest_int('min_samples_leaf', 1, 50)
    }
    clf = GradientBoostingClassifier(**params, random_state=42)
    return cross_val_score(clf, X_train, y_train, cv=3, scoring='roc_auc').mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, n_jobs=-1)
print(study.best_params, study.best_value)
```
Unlike Random Search (which samples blindly), Optuna uses the *results of previous trials* to intelligently decide which parameter combinations to try next — it builds a probabilistic model of "where good scores tend to be" and focuses search there. This is why it can find good hyperparameters with fewer total trials than random search, especially in complex/expensive search spaces.

---

## Summary table (from the notes)

| Algorithm | When to use | Key hyperparameter |
|---|---|---|
| K-Means | Large data, convex clusters | k (use silhouette) |
| HAC | Small data, hierarchy matters | linkage (use ward) |
| DBSCAN | Arbitrary shapes, detect outliers | eps, min_samples |
| GMM | Soft assignments, elliptical shapes | n_components (via BIC) |
| GridSearch | Small, discrete param space | cv, scoring |
| RandomizedSearch | Large/continuous space | n_iter |
| Optuna | Complex, expensive models | n_trials |

## Worked example: manual K-Means assignment

This is a good sanity check for the E-step math. Given centroids **C1 = (0,0)** and **C2 = (5,5)**, and point **P = (2,2)**:

1. Squared distance to C1: (2-0)² + (2-0)² = 4 + 4 = **8**
2. Squared distance to C2: (2-5)² + (2-5)² = 9 + 9 = **18**
3. Since 8 < 18, P is assigned to **Cluster 1** (E-step)
4. M-step: if P were the *only* point in Cluster 1, the new centroid C1 becomes exactly P's position, **(2,2)** — because the centroid is just the mean of its assigned points

---

Want me to turn any of this into a quiz to test yourself, or work through a full end-to-end example on a real dataset (e.g., picking k with silhouette, then comparing K-Means vs DBSCAN vs GMM on the same data)?
