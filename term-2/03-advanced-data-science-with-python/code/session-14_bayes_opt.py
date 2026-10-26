"""Bayesian optimisation with a Gaussian-process surrogate and expected improvement, vs random search, tuning an SVM."""
import numpy as np
from scipy.stats import norm
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

X, y = load_breast_cancer(return_X_y=True)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
def score(log_c, log_g):                       # the expensive black box we want to maximise
    model = make_pipeline(StandardScaler(), SVC(C=10 ** log_c, gamma=10 ** log_g))
    return cross_val_score(model, X, y, cv=cv).mean()
bounds = np.array([[-2, 3], [-5, 0]])          # log10 C, log10 gamma
rng = np.random.default_rng(0)
BUDGET = 15

# Random search
rand_pts = rng.uniform(bounds[:, 0], bounds[:, 1], (BUDGET, 2))
rand_scores = [score(*p) for p in rand_pts]

# Bayesian optimisation
pts = list(rng.uniform(bounds[:, 0], bounds[:, 1], (4, 2)))   # 4 random starting points
scores = [score(*p) for p in pts]
gp = GaussianProcessRegressor(Matern(nu=2.5) + WhiteKernel(1e-4), normalize_y=True, random_state=0)
while len(pts) < BUDGET:
    gp.fit(np.array(pts), np.array(scores))
    cand = rng.uniform(bounds[:, 0], bounds[:, 1], (2000, 2))
    mu, sd = gp.predict(cand, return_std=True)
    best = max(scores)
    z = (mu - best) / np.maximum(sd, 1e-9)
    ei = (mu - best) * norm.cdf(z) + sd * norm.pdf(z)          # expected improvement
    nxt = cand[np.argmax(ei)]
    pts.append(nxt); scores.append(score(*nxt))
    print(f"trial {len(pts):>2}: log10 C = {nxt[0]:5.2f}, log10 gamma = {nxt[1]:5.2f} -> CV accuracy {scores[-1]:.4f}")

print(f"\nbest after {BUDGET} trials | random search: {max(rand_scores):.4f} | Bayesian optimisation: {max(scores):.4f}")
print("best-so-far curve (random):  ", np.round(np.maximum.accumulate(rand_scores)[[3, 7, 11, 14]], 4))
print("best-so-far curve (Bayesian):", np.round(np.maximum.accumulate(scores)[[3, 7, 11, 14]], 4))
