"""K-means, agglomerative, DBSCAN and Gaussian mixture on moons and blobs: who copes with which shape?"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

moons, ym = make_moons(400, noise=.06, random_state=0)
blobs, yb = make_blobs(400, centers=[[0, 0], [4, 0], [2, 3]], cluster_std=[.4, 1.2, .6], random_state=0)
datasets = {"moons": (StandardScaler().fit_transform(moons), ym), "blobs": (StandardScaler().fit_transform(blobs), yb)}
algos = {
    "K-means": lambda k: KMeans(k, n_init=10, random_state=0),
    "Agglomerative (single)": lambda k: AgglomerativeClustering(k, linkage="single"),
    "DBSCAN": lambda k: DBSCAN(eps=0.3, min_samples=5),
    "Gaussian mixture": lambda k: GaussianMixture(k, random_state=0),
}
fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
print(f"{'algorithm':<24}{'moons ARI':>10}{'blobs ARI':>10}")
scores = {a: [] for a in algos}
for r, (dname, (X, y)) in enumerate(datasets.items()):
    k = len(set(y))
    for c, (aname, make) in enumerate(algos.items()):
        labels = make(k).fit_predict(X)
        scores[aname].append(adjusted_rand_score(y, labels))
        axes[r, c].scatter(X[:, 0], X[:, 1], c=labels, cmap="plasma", s=8)
        axes[r, c].set_title(f"{aname} on {dname}", fontsize=9); axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
for a, s in scores.items():
    print(f"{a:<24}{s[0]:>10.2f}{s[1]:>10.2f}")
fig.tight_layout(); fig.savefig("session-06_cluster_families.png", dpi=100); plt.show()
