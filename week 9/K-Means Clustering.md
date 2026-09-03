# K-Means Clustering

**Algorithm**

1. Initialise $k$ centroids (random or k-means++ smart init)
2. E-step: Assign each point to nearest centroid
3. M-step: Recompute each centroid as mean of assigned points
4. Repeat E/M until convergence (centroids stop moving)

Objective: minimise Within-Cluster Sum of Squares (WCSS / Inertia)


$$WCSS = \sum_{i} \sum_{x \in C_i} \Vert{}x - \mu_i\Vert{}^2$$

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ALWAYS scale before clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=5,
    init='k-means++',  # smart init (default, avoids bad random starts)
    n_init=10,         # run 10 times, pick best WCSS
    max_iter=300,
    tol=1e-4,
    random_state=42
)
labels = kmeans.fit_predict(X_scaled)
print(f"Inertia (WCSS): {kmeans.inertia_:.1f}")
print(f"Cluster sizes: {pd.Series(labels).value_counts().sort_index().values}")
print(f"Centroids shape: {kmeans.cluster_centers_.shape}")

# Add cluster labels to dataframe
df['cluster'] = labels

```

**Finding Optimal k**

```python
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

inertias = []
sil_scores = []
db_scores = []
ch_scores = []
K_range = range(2, 15)

for k in K_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    labels_k = km.labels_
    sil_scores.append(silhouette_score(X_scaled, labels_k))
    db_scores.append(davies_bouldin_score(X_scaled, labels_k))
    ch_scores.append(calinski_harabasz_score(X_scaled, labels_k))

# Elbow: look for "knee" in inertia plot
# Silhouette: higher is better (max = 1)
# Davies-Bouldin: lower is better
# Calinski-Harabasz: higher is better
best_k_sil = list(K_range)[np.argmax(sil_scores)]
print(f"Best k by Silhouette: {best_k_sil}")

```

**Clustering Metrics Reference**

| Metric | Range | Better when | Without ground truth? |
| --- | --- | --- | --- |
| Silhouette Score | -1 to 1 | Higher | ✓ (internal) |
| Davies-Bouldin | $\ge 0$ | Lower | ✓ (internal) |
| Calinski-Harabasz | $\ge 0$ | Higher | ✓ (internal) |
| Inertia (WCSS) | $\ge 0$ | Lower | ✓ (elbow) |
| ARI | -1 to 1 | Higher | ✗ (needs labels) |
| NMI | 0 to 1 | Higher | ✗ (needs labels) |

---

* **2. Hierarchical Agglomerative Clustering (HAC)**

**Algorithm**
Bottom-up approach:

1. Start: every point is its own cluster
2. Merge the two closest clusters
3. Repeat until one cluster remains
4. Cut dendrogram at desired level to get $k$ clusters

Linkage criteria (how to measure cluster-cluster distance):

* `single`: minimum distance between any two points (chain-like)
* `complete`: maximum distance (compact)
* `average`: average pairwise distance (balanced)
* `ward`: minimise within-cluster variance (usually best)

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Dendrogram (on a subsample for large data)
linkage_matrix = linkage(X_scaled[:200], method='ward')
plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix, truncate_mode='lastp', p=20, leaf_rotation=90, leaf_font_size=10)
plt.axhline(y=10, color='r', linestyle='--')  # cut level
plt.title('Dendrogram'); plt.show()

# Fit with chosen k
agg = AgglomerativeClustering(
    n_clusters=5,
    linkage='ward',      # 'ward', 'complete', 'average', 'single'
    metric='euclidean'   # 'euclidean', 'cosine', 'manhattan'
)
labels = agg.fit_predict(X_scaled)

```

**HAC Linkage Comparison**

| Linkage | Cluster shape | Sensitive to outliers | Recommended |
| --- | --- | --- | --- |
| ward | Compact, equal | No | Yes, default |
| complete | Compact | Yes | Yes |
| average | Balanced | Moderate | Yes |
| single | Chain-like | Very | Rarely |

---

* **3. DBSCAN**

**Algorithm**
Parameters: $\epsilon$ (epsilon) = neighbourhood radius, `min_samples` = core point threshold

* Core point: has $\ge \text{min samples}$ within $\epsilon$
* Border point: within $\epsilon$ of a core point but not core itself
* Noise point: not reachable from any core point (label = -1)
* Strengths: finds arbitrary shapes, detects outliers automatically
* Weaknesses: struggles with varying densities, hard to set $\epsilon$

```python
from sklearn.cluster import DBSCAN

db = DBSCAN(
    eps=0.5,         # neighbourhood radius (critical param)
    min_samples=5,   # core point threshold
    metric='euclidean',
    n_jobs=-1
)
labels = db.fit_predict(X_scaled)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = np.sum(labels == -1)
print(f"Clusters: {n_clusters}, Noise: {n_noise}")

# Choosing eps: k-distance plot
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=5)
nn.fit(X_scaled)
distances, _ = nn.kneighbors(X_scaled)
distances = np.sort(distances[:, -1])
plt.plot(distances)
plt.xlabel('Points sorted by distance')
plt.ylabel('5-NN distance')
# Look for the "elbow" → that is the eps value

```

**Clustering Algorithm Comparison**

| Algorithm | Shapes | Noise | $k$ needed | Scales |
| --- | --- | --- | --- | --- |
| K-Means | Convex | No | Yes | ✓✓✓ |
| HAC | Any | No | Yes | ✗ (large data) |
| DBSCAN | Any | Yes | No ($\epsilon$, min) | ✓ |
| HDBSCAN | Any | Yes | No | ✓✓ |
| Gaussian Mixture | Elliptical | Soft | Yes | ✓ |

---

* **4. Gaussian Mixture Models (GMM)**

```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(
    n_components=5,
    covariance_type='full',  # 'full', 'tied', 'diag', 'spherical'
    n_init=3,
    random_state=42
)
gmm.fit(X_scaled)
labels = gmm.predict(X_scaled)
probs = gmm.predict_proba(X_scaled)  # soft assignments!

# Select n_components using BIC/AIC
bics = [GaussianMixture(n_components=k, random_state=42).fit(X_scaled).bic(X_scaled) for k in range(1, 15)]
best_k = np.argmin(bics) + 1

```

---

* **5. Hyperparameter Tuning: Complete Guide**

**Search Strategy Decision**

* $n_{\text{hyperparameters}} \le 3$, discrete values $\rightarrow$ `GridSearchCV`
* $n_{\text{hyperparameters}} > 3$ or continuous $\rightarrow$ `RandomizedSearchCV`
* Large search space, expensive evaluation $\rightarrow$ Bayesian Optimisation (`optuna`, `hyperopt`)

```python
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV,
    HalvingGridSearchCV, HalvingRandomSearchCV
)
from scipy.stats import uniform, randint, loguniform

# ── GridSearchCV ──────────────────────────────────────────────
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 5, 7, None],
    'min_samples_leaf': [1, 5, 10],
    'max_features': ['sqrt', 'log2', 0.5]
}
# Total combinations: 3 × 4 × 3 × 3 = 108 × 5 CV folds = 540 fits!
gs = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    refit=True,  # refit best model on full train set
    verbose=1,
    return_train_score=True
)
gs.fit(X_train, y_train)
print(gs.best_params_, gs.best_score_)

# ── RandomizedSearchCV ────────────────────────────────────────
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
    n_iter=100,  # number of random combinations to try
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42
)
rs.fit(X_train, y_train)

# ── Successive Halving (faster) ───────────────────────────────
from sklearn.model_selection import HalvingRandomSearchCV
hs = HalvingRandomSearchCV(
    RandomForestClassifier(), param_dist,
    factor=3, cv=5, n_jobs=-1, random_state=42
)
hs.fit(X_train, y_train)

```

**Concept Breakdown: The n_iter vs cv Trap**

* **Problem:** What exactly do `n_iter` and `cv` control in `RandomizedSearchCV`?
* **Reasoning:** People often confuse these two. `n_iter` is the number of **random parameter combinations** the search will try. `cv` is the number of **cross-validation folds** trained **for each combination**.
* **Result:** The total number of model fits is $\text{n iter} \times \text{cv}$.
* **Tip:** Increasing `n_iter` explores more of the hyperparameter space. Increasing `cv` gives a more reliable score estimate for each point in that space.

**Bayesian Optimisation with Optuna**

```python
# pip install optuna
import optuna
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 8),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 50)
    }
    clf = GradientBoostingClassifier(**params, random_state=42)
    return cross_val_score(clf, X_train, y_train, cv=3, scoring='roc_auc').mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, n_jobs=-1)
print(study.best_params, study.best_value)

```

---

**Key Takeaways**

| Algorithm | When to use | Key hyperparameter |
| --- | --- | --- |
| K-Means | Large data, convex clusters | $k$ (use silhouette) |
| HAC | Small data, hierarchy matters | linkage (use ward) |
| DBSCAN | Arbitrary shapes, detect outliers | $\text{eps}$, $\text{min samples}$ |
| GMM | Soft assignments, elliptical | `n_components` (BIC) |
| GridSearch | Small, discrete param space | `cv`, `scoring` |
| RandomizedSearch | Large/continuous space | `n_iter` |
| Optuna | Complex, expensive models | `n_trials` |

**Pen & Paper Example: K-Means Manual Assignment**

* **Input:** Centroids $C_1 = (0,0)$, $C_2 = (5,5)$. Point $P = (2,2)$.
* **1. Distance to $C_1$:** $(2-0)^2 + (2-0)^2 = 4 + 4 = 8$ (squared Euclidean).
* **2. Distance to $C_2$:** $(2-5)^2 + (2-5)^2 = 9 + 9 = 18$.
* **3. Assignment (E-step):** Point $P$ is assigned to Cluster 1.
* **4. Update (M-step):** If $P$ was the only point assigned to $C_1$, the new $C_1$ becomes exactly $(2,2)$.
