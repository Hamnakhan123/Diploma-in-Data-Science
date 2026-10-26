"""PCA from scratch: standardise, covariance, eigen-decomposition, projection; compare with scikit-learn."""
import numpy as np
from sklearn.decomposition import PCA

rng = np.random.default_rng(0)
# 200 students: maths, physics and computing marks are correlated; art is independent
ability = rng.normal(0, 1, 200)
X = np.c_[70 + 10 * ability + rng.normal(0, 4, 200),
          68 + 9 * ability + rng.normal(0, 4, 200),
          72 + 8 * ability + rng.normal(0, 5, 200),
          65 + rng.normal(0, 10, 200)]
cols = ["maths", "physics", "computing", "art"]

Xs = (X - X.mean(0)) / X.std(0)                      # 1. standardise
C = np.cov(Xs, rowvar=False)                          # 2. covariance matrix (4 x 4)
eigval, eigvec = np.linalg.eigh(C)                    # 3. eigen-decomposition (ascending)
order = eigval.argsort()[::-1]
eigval, eigvec = eigval[order], eigvec[:, order]      # 4. sort largest first
ratio = eigval / eigval.sum()
print("explained variance ratio:", ratio.round(3))
print("cumulative:              ", ratio.cumsum().round(3))
print("\nloadings (columns = PC1, PC2):")
for name, row in zip(cols, eigvec[:, :2]):
    print(f"  {name:<10}{row[0]:7.2f}{row[1]:7.2f}")
Z = Xs @ eigvec[:, :2]                                # 5. project onto 2 components
print("\nfirst 3 students in PC space:\n", Z[:3].round(2))

sk = PCA(2).fit(Xs)
print("\nscikit-learn ratio:", sk.explained_variance_ratio_.round(3), "(same values; signs of PCs may flip)")
