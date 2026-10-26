"""Real-life problem: reduce a 30-question survey to a few attitude scores with PCA, then map with t-SNE."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

rng = np.random.default_rng(5)
n = 1500
themes = {"digital": 8, "branch": 8, "fees": 7, "staff": 7}             # questions per hidden attitude
groups = rng.choice(3, n, p=[.45, .35, .20])                            # young digital, traditional, unhappy
means = {0: {"digital": 1.0, "branch": -0.6, "fees": 0.2, "staff": 0.1},
         1: {"digital": -0.8, "branch": 0.9, "fees": 0.1, "staff": 0.8},
         2: {"digital": -0.3, "branch": -0.5, "fees": -1.2, "staff": -0.9}}
cols, data = [], []
for theme, q in themes.items():
    latent = np.array([means[g][theme] for g in groups]) + rng.normal(0, .6, n)
    for i in range(q):
        ans = np.clip(np.round(3 + 0.9 * latent + rng.normal(0, .7, n)), 1, 5)
        data.append(ans); cols.append(f"{theme}_q{i + 1}")
survey = pd.DataFrame(np.array(data).T, columns=cols)
print("survey shape:", survey.shape)

Xs = StandardScaler().fit_transform(survey)
pca = PCA().fit(Xs)
print("explained variance of first 6 PCs:", pca.explained_variance_ratio_[:6].round(3))
k = int((pca.explained_variance_ > 1).sum())                           # Kaiser rule: eigenvalue > 1
print(f"components with eigenvalue > 1 (Kaiser rule): {k}, together {pca.explained_variance_ratio_[:k].sum():.1%}")

pca_k = PCA(k, random_state=0).fit(Xs)
load = pd.DataFrame(pca_k.components_.T, index=cols, columns=[f"PC{i + 1}" for i in range(k)])
theme_load = load.groupby(lambda c: c.split("_")[0]).mean().round(2)
print("\naverage loading of each question theme on each component:\n", theme_load.to_string())

scores = pd.DataFrame(pca_k.transform(Xs), columns=load.columns)
scores["segment"] = KMeans(3, n_init=10, random_state=0).fit_predict(scores)
print("\nmean component score per segment:\n", scores.groupby("segment").mean().round(2).to_string())

Z = TSNE(2, perplexity=40, init="pca", learning_rate="auto", random_state=0).fit_transform(Xs)
plt.figure(figsize=(6.5, 5))
plt.scatter(Z[:, 0], Z[:, 1], c=scores["segment"], cmap="plasma", s=6)
plt.title("t-SNE map of 1,500 survey respondents (colour = K-means segment on PC scores)")
plt.xticks([]); plt.yticks([]); plt.tight_layout(); plt.savefig("session-08_survey_map.png", dpi=110); plt.show()
