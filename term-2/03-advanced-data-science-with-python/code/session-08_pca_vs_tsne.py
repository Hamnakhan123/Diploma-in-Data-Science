"""Compare a 2-D PCA projection with a t-SNE embedding of the digits and measure neighbourhood preservation."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

X, y = load_digits(return_X_y=True)
Z_pca = PCA(2, random_state=0).fit_transform(X)
X50 = PCA(30, random_state=0).fit_transform(X)                 # speed-up trick: PCA first
Z_tsne = TSNE(2, perplexity=30, init="pca", learning_rate="auto", random_state=0).fit_transform(X50)

for name, Z in [("PCA", Z_pca), ("t-SNE", Z_tsne)]:
    tw = trustworthiness(X, Z, n_neighbors=10)
    knn = cross_val_score(KNeighborsClassifier(5), Z, y, cv=5).mean()
    print(f"{name:<6} trustworthiness@10 = {tw:.3f}   5-NN accuracy on the 2-D map = {knn:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for a, Z, t in [(ax[0], Z_pca, "PCA (linear): digits overlap"), (ax[1], Z_tsne, "t-SNE (perplexity 30): clear groups")]:
    sc = a.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=6)
    for d in range(10):
        cx, cy = np.median(Z[y == d], axis=0); a.text(cx, cy, str(d), fontsize=13, weight="bold")
    a.set_title(t); a.set_xticks([]); a.set_yticks([])
fig.tight_layout(); fig.savefig("session-08_pca_vs_tsne.png", dpi=100); plt.show()
