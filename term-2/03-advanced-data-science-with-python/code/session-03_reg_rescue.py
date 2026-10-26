"""40 rows and 60 features (only 5 useful): compare OLS, Ridge, Lasso and ElasticNet with CV-chosen alpha."""
import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

rng = np.random.default_rng(42)
n, p = 120, 60
X = rng.normal(size=(n, p))
true_w = np.zeros(p); true_w[:5] = [8, -6, 5, 4, -3]          # only the first 5 matter
y = X @ true_w + rng.normal(0, 4, n)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, train_size=40, random_state=0)

alphas = np.logspace(-3, 3, 30)
models = {
    "OLS (no penalty)": LinearRegression(),
    "RidgeCV": RidgeCV(alphas=alphas),
    "LassoCV": LassoCV(alphas=alphas, cv=5, max_iter=50000, random_state=0),
    "ElasticNetCV": ElasticNetCV(alphas=alphas, l1_ratio=[.2, .5, .8], cv=5, max_iter=50000, random_state=0),
}
print(f"{'model':<18}{'train R2':>9}{'test R2':>9}{'test RMSE':>11}  non-zero coefs  alpha")
for name, m in models.items():
    pipe = make_pipeline(StandardScaler(), m).fit(X_tr, y_tr)
    est = pipe[-1]
    nz = int((np.abs(est.coef_) > 1e-6).sum())
    a = getattr(est, "alpha_", "-")
    print(f"{name:<18}{r2_score(y_tr, pipe.predict(X_tr)):>9.2f}{r2_score(y_te, pipe.predict(X_te)):>9.2f}"
          f"{mean_squared_error(y_te, pipe.predict(X_te)) ** .5:>11.2f}  {nz:>14}  {a if a == '-' else round(a, 3)}")
lasso = make_pipeline(StandardScaler(), models["LassoCV"]).fit(X_tr, y_tr)[-1]
print("\nLasso's largest coefficients at positions:", np.argsort(-np.abs(lasso.coef_))[:5])
