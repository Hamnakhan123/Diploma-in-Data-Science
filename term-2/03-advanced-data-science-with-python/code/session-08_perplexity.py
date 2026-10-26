"""t-SNE with perplexity 2, 10, 30 and 100 on three clusters of very different sizes and spreads."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

rng = np.random.default_rng(0)
# 10-D data: one tight cluster (50), one wide cluster (200), and one mid cluster (100)
A = rng.normal(0, .3, (50, 10)); B = rng.normal(5, 2.0, (200, 10)); C = rng.normal(-5, 1.0, (100, 10))
X = np.vstack([A, B, C]); y = np.repeat([0, 1, 2], [50, 200, 100])
print("true spread (mean distance to own centre):",
      [round(float(np.linalg.norm(g - g.mean(0), axis=1).mean()), 2) for g in (A, B, C)])

fig, ax = plt.subplots(1, 4, figsize=(16, 4))
for a, perp in zip(ax, [2, 10, 30, 100]):
    Z = TSNE(2, perplexity=perp, init="pca", learning_rate="auto", random_state=1).fit_transform(X)
    spread = [round(float(np.linalg.norm(Z[y == k] - Z[y == k].mean(0), axis=1).mean()), 1) for k in range(3)]
    print(f"perplexity {perp:>3}: spread on the map (tight, wide, mid) = {spread}")
    a.scatter(Z[:, 0], Z[:, 1], c=y, cmap="plasma", s=8); a.set_title(f"perplexity = {perp}"); a.set_xticks([]); a.set_yticks([])
fig.suptitle("Same data, different perplexity: cluster sizes on a t-SNE map are not real sizes")
fig.tight_layout(); fig.savefig("session-08_perplexity.png", dpi=100); plt.show()

Z1 = TSNE(2, perplexity=30, random_state=1, init="random").fit_transform(X)
Z2 = TSNE(2, perplexity=30, random_state=7, init="random").fit_transform(X)
print("\ntwo random seeds give different coordinates:", Z1[0].round(1), "vs", Z2[0].round(1),
      "(the grouping is the same, the layout is not)")
