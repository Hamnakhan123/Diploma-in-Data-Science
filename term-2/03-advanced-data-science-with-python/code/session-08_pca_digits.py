"""PCA on 8x8 digit images: scree plot, components for 95% variance, and image reconstruction quality."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

X, y = load_digits(return_X_y=True)
pca = PCA().fit(X)
cum = pca.explained_variance_ratio_.cumsum()
for thr in [.5, .8, .9, .95, .99]:
    print(f"{thr:.0%} of variance needs {np.argmax(cum >= thr) + 1:>2} of 64 components")

fig, ax = plt.subplots(2, 6, figsize=(13, 5))
ax0 = plt.subplot2grid((2, 6), (0, 0), colspan=3)
ax0.bar(range(1, 31), pca.explained_variance_ratio_[:30], color="#FF6B1A", label="per component")
ax0.plot(range(1, 31), cum[:30], color="#A78BFA", marker="o", ms=3, label="cumulative")
ax0.set_title("Scree plot (first 30 components)"); ax0.set_xlabel("component"); ax0.legend()
for a in ax[0, :3].ravel(): a.remove()
for j, k in enumerate([2, 10, 40]):
    p = PCA(k).fit(X)
    rec = p.inverse_transform(p.transform(X[:1]))
    err = np.mean((X - p.inverse_transform(p.transform(X))) ** 2)
    print(f"k={k:>2}: mean squared reconstruction error = {err:6.2f}")
    ax[0, 3 + j].imshow(rec.reshape(8, 8), cmap="gray_r"); ax[0, 3 + j].set_title(f"{k} components"); ax[0, 3 + j].axis("off")
for j in range(6):
    ax[1, j].imshow(pca.components_[j].reshape(8, 8), cmap="RdBu"); ax[1, j].set_title(f"PC{j + 1}"); ax[1, j].axis("off")
fig.suptitle("Top: scree plot and a '0' rebuilt from 2, 10, 40 components. Bottom: the first six components as images")
fig.tight_layout(); fig.savefig("session-08_pca_digits.png", dpi=100); plt.show()
