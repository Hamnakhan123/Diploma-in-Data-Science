"""Elbow (inertia) and silhouette analysis to choose k for K-means; saves a two-panel chart."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

X, _ = make_blobs(n_samples=600, centers=4, cluster_std=1.1, random_state=7)
X = StandardScaler().fit_transform(X)
ks = range(2, 10)
inertia, sil = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    inertia.append(km.inertia_); sil.append(silhouette_score(X, km.labels_))
    print(f"k={k}: inertia={km.inertia_:7.1f}  silhouette={sil[-1]:.3f}")
print("best k by silhouette:", list(ks)[int(np.argmax(sil))])

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].plot(list(ks), inertia, "o-", color="#FF6B1A"); ax[0].set_title("Elbow method"); ax[0].set_xlabel("k"); ax[0].set_ylabel("inertia")
ax[1].plot(list(ks), sil, "s-", color="#A78BFA"); ax[1].set_title("Silhouette score"); ax[1].set_xlabel("k")
fig.tight_layout(); fig.savefig("session-06_choose_k.png", dpi=110); plt.show()
