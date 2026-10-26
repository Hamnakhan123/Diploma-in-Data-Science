"""Choose eps for DBSCAN with a k-distance plot, then see how eps changes clusters and noise."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

rng = np.random.default_rng(0)
moons, _ = make_moons(500, noise=.06, random_state=0)
blob, _ = make_blobs(150, centers=[[2.5, 1.8]], cluster_std=.15, random_state=0)
noise = rng.uniform([-1.5, -1], [3.5, 2.5], (40, 2))
X = StandardScaler().fit_transform(np.vstack([moons, blob, noise]))

k = 5
dist, _ = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(X)
kdist = np.sort(dist[:, -1])
knee = kdist[int(len(kdist) * .95)]
print(f"95th percentile of {k}-NN distance (near the knee): {knee:.3f}")

print("\n  eps  clusters  noise points  core points")
for eps in [0.08, 0.15, knee, 0.4, 0.8]:
    db = DBSCAN(eps=eps, min_samples=k).fit(X)
    n_cl = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    print(f"{eps:5.2f}  {n_cl:>8}  {(db.labels_ == -1).sum():>12}  {len(db.core_sample_indices_):>11}")

db = DBSCAN(eps=knee, min_samples=k).fit(X)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(kdist, color="#FF6B1A"); ax[0].axhline(knee, ls="--", color="grey")
ax[0].set_title(f"k-distance plot (k={k}): pick eps at the knee"); ax[0].set_xlabel("points sorted by distance"); ax[0].set_ylabel(f"distance to {k}th neighbour")
ax[1].scatter(X[:, 0], X[:, 1], c=db.labels_, cmap="plasma", s=10)
ax[1].scatter(X[db.labels_ == -1, 0], X[db.labels_ == -1, 1], c="black", marker="x", s=20, label="noise (-1)")
ax[1].set_title(f"DBSCAN eps={knee:.2f}: 3 clusters + noise"); ax[1].legend()
fig.tight_layout(); fig.savefig("session-07_dbscan.png", dpi=110); plt.show()
