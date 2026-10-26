"""How Ridge and Lasso coefficients shrink as alpha grows (diabetes data, 10 features)."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler

data = load_diabetes()
X = StandardScaler().fit_transform(data.data); y = data.target
names = data.feature_names
alphas = np.logspace(-2, 4, 60)
ridge = np.array([Ridge(alpha=a).fit(X, y).coef_ for a in alphas])
lasso = np.array([Lasso(alpha=a / 100, max_iter=20000).fit(X, y).coef_ for a in alphas])

ols = LinearRegression().fit(X, y).coef_
print("feature   OLS     Ridge(a=100)  Lasso(a=5)")
r100 = Ridge(alpha=100).fit(X, y).coef_; l5 = Lasso(alpha=5).fit(X, y).coef_
for n, a, b, c in zip(names, ols, r100, l5):
    print(f"{n:<6}{a:>8.1f}{b:>12.1f}{c:>12.1f}")
print("Lasso(alpha=5) keeps", int((l5 != 0).sum()), "of 10 features:", [n for n, c in zip(names, l5) if c != 0])

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for j in range(X.shape[1]):
    ax[0].plot(alphas, ridge[:, j], label=names[j]); ax[1].plot(alphas / 100, lasso[:, j])
ax[0].set_xscale("log"); ax[1].set_xscale("log")
ax[0].set_title("Ridge: coefficients shrink smoothly"); ax[1].set_title("Lasso: coefficients hit exactly zero")
for a in ax: a.set_xlabel("alpha (penalty strength)"); a.axhline(0, color="grey", lw=.8)
ax[0].set_ylabel("coefficient"); ax[0].legend(fontsize=7, ncol=2)
fig.tight_layout(); fig.savefig("session-03_regularisation_paths.png", dpi=110); plt.show()
