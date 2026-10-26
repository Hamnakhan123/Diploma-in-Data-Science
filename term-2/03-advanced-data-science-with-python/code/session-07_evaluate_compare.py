"""Internal and external clustering metrics for K-means, DBSCAN and Mean Shift on blobs and moons."""
import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, MeanShift, estimate_bandwidth
from sklearn.metrics import (silhouette_score, davies_bouldin_score, calinski_harabasz_score,
                             adjusted_rand_score, normalized_mutual_info_score)

data = {"blobs": make_blobs(600, centers=4, cluster_std=.9, random_state=3),
        "moons": make_moons(600, noise=.06, random_state=3)}
print(f"{'data':<6}{'algorithm':<11}{'k found':>8}{'silh':>7}{'DB':>7}{'CH':>8}{'ARI':>7}{'NMI':>7}")
for dname, (X, y) in data.items():
    X = StandardScaler().fit_transform(X)
    bw = estimate_bandwidth(X, quantile=.2, random_state=0)
    algos = {"K-means": KMeans(len(set(y)), n_init=10, random_state=0),
             "DBSCAN": DBSCAN(eps=.25, min_samples=5),
             "MeanShift": MeanShift(bandwidth=bw)}
    for aname, algo in algos.items():
        lab = algo.fit_predict(X)
        mask = lab != -1                                       # internal metrics ignore DBSCAN noise
        k = len(set(lab[mask]))
        sil = silhouette_score(X[mask], lab[mask]) if k > 1 else float("nan")
        dbi = davies_bouldin_score(X[mask], lab[mask]) if k > 1 else float("nan")
        ch = calinski_harabasz_score(X[mask], lab[mask]) if k > 1 else float("nan")
        print(f"{dname:<6}{aname:<11}{k:>8}{sil:>7.2f}{dbi:>7.2f}{ch:>8.0f}"
              f"{adjusted_rand_score(y, lab):>7.2f}{normalized_mutual_info_score(y, lab):>7.2f}")
