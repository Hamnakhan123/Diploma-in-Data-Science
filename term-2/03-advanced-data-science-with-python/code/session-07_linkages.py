"""Ward, complete, average and single linkage on three data shapes, scored with ARI."""
import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

shapes = {
    "blobs": make_blobs(500, centers=3, cluster_std=1.0, random_state=1),
    "moons": make_moons(500, noise=.05, random_state=1),
    "circles": make_circles(500, noise=.04, factor=.5, random_state=1),
}
print(f"{'linkage':<10}" + "".join(f"{s:>10}" for s in shapes))
for link in ["ward", "complete", "average", "single"]:
    row = f"{link:<10}"
    for name, (X, y) in shapes.items():
        lab = AgglomerativeClustering(n_clusters=len(set(y)), linkage=link).fit_predict(StandardScaler().fit_transform(X))
        row += f"{adjusted_rand_score(y, lab):>10.2f}"
    print(row)
