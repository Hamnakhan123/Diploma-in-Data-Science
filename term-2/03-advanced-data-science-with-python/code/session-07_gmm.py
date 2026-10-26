"""Elongated clusters: K-means vs Gaussian mixture, choosing components by BIC, soft probabilities."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

rng = np.random.default_rng(2)
# Three stretched, tilted groups (e.g. commuter trips along three roads)
def stretched(center, angle, n):
    pts = rng.normal(size=(n, 2)) * [2.5, .35]
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return pts @ R.T + center
X = np.vstack([stretched([0, 0], .6, 300), stretched([4, -3], .6, 300), stretched([1, 4], -.4, 300)])
y = np.repeat([0, 1, 2], 300)

km = KMeans(3, n_init=10, random_state=0).fit(X)
print(f"K-means ARI: {adjusted_rand_score(y, km.labels_):.3f}")
print("\ncomponents  BIC (full covariance)")
bics = {}
for k in range(1, 7):
    bics[k] = GaussianMixture(k, covariance_type="full", random_state=0).fit(X).bic(X)
    print(f"{k:>10}  {bics[k]:10.1f}")
best = min(bics, key=bics.get)
gmm = GaussianMixture(best, covariance_type="full", random_state=0).fit(X)
print(f"best by BIC: {best} components | GMM ARI: {adjusted_rand_score(y, gmm.predict(X)):.3f}")

for ctype in ["spherical", "diag", "tied", "full"]:
    g = GaussianMixture(3, covariance_type=ctype, random_state=0).fit(X)
    print(f"covariance_type={ctype:<9}  ARI {adjusted_rand_score(y, g.predict(X)):.3f}  BIC {g.bic(X):8.1f}")

probs = gmm.predict_proba(X)
unsure = (probs.max(axis=1) < .8).sum()
print(f"\npoints with less than 80% membership in any component: {unsure}")
print("example soft membership:", probs[np.argmin(probs.max(axis=1))].round(2))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].scatter(X[:, 0], X[:, 1], c=km.labels_, cmap="plasma", s=6); ax[0].set_title("K-means: round clusters cut across roads")
ax[1].scatter(X[:, 0], X[:, 1], c=gmm.predict(X), cmap="plasma", s=6); ax[1].set_title("GMM (full covariance): follows the ovals")
fig.tight_layout(); fig.savefig("session-07_gmm_vs_kmeans.png", dpi=110); plt.show()
