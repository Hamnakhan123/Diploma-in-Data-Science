"""Mini project: compress a 40-feature product catalogue with PCA, find similar products, map it with t-SNE."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

rng = np.random.default_rng(8)
categories = ["phones", "laptops", "headphones", "smartwatches"]
singular = ["phone", "laptop", "headphone", "smartwatch"]
n_per = 150
centres = rng.normal(0, 3, (4, 6))                       # 6 hidden product properties per category
latent = np.vstack([c + rng.normal(0, 1, (n_per, 6)) for c in centres])
mixing = rng.normal(0, 1, (6, 40))                       # 40 measured attributes mix the hidden ones
X = latent @ mixing + rng.normal(0, 1.5, (len(latent), 40))
cat = np.repeat(categories, n_per)
names = [f"{c}-{i:03d}" for c in singular for i in range(n_per)]

Xs = StandardScaler().fit_transform(X)
pca = PCA(0.90, random_state=0).fit(Xs)
Z = pca.transform(Xs)
print(f"40 attributes -> {pca.n_components_} components keep {pca.explained_variance_ratio_.sum():.1%} of variance")
print(f"storage: {X.size:,} numbers -> {Z.size:,} numbers ({Z.size / X.size:.0%})")

nn = NearestNeighbors(n_neighbors=4).fit(Z)
for q in [0, 160, 470]:
    idx = nn.kneighbors(Z[q:q + 1], return_distance=False)[0][1:]
    print(f"customers who viewed {names[q]} may like: {[names[i] for i in idx]}")

emb = TSNE(2, perplexity=30, init="pca", learning_rate="auto", random_state=0).fit_transform(Z)
plt.figure(figsize=(6.5, 5))
for c in categories:
    m = cat == c; plt.scatter(emb[m, 0], emb[m, 1], s=8, label=c)
plt.legend(); plt.xticks([]); plt.yticks([]); plt.title("Product catalogue map (PCA then t-SNE)")
plt.tight_layout(); plt.savefig("session-08_mini_project_catalogue.png", dpi=110); plt.show()
