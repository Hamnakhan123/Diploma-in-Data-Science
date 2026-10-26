"""Global explanations for a gradient-boosting model: permutation importance and partial dependence plots."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

rng = np.random.default_rng(0)
n = 3000
X = pd.DataFrame({"area_sqft": rng.normal(1300, 400, n).clip(400, 3500), "age_years": rng.integers(0, 40, n),
                  "distance_km": rng.exponential(6, n).clip(0.5, 30), "floor": rng.integers(0, 15, n),
                  "random_noise": rng.normal(0, 1, n)}).astype(float)    # float dtypes for partial dependence
y = (0.09 * X.area_sqft - 1.2 * X.age_years - 25 * np.log(X.distance_km) + 3 * np.minimum(X.floor, 8)
     + rng.normal(0, 12, n))                                   # price in lakh (synthetic)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.25, random_state=0)
model = HistGradientBoostingRegressor(random_state=0).fit(X_tr, y_tr)
print(f"test R2: {model.score(X_te, y_te):.3f}")

imp = permutation_importance(model, X_te, y_te, n_repeats=10, random_state=0, scoring="r2")
order = imp.importances_mean.argsort()[::-1]
print("\npermutation importance (drop in R2 when shuffled):")
for j in order:
    print(f"  {X.columns[j]:<13} {imp.importances_mean[j]:.3f} ± {imp.importances_std[j]:.3f}")

fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
PartialDependenceDisplay.from_estimator(model, X_te, ["area_sqft", "distance_km", "floor"], kind="both",
                                        subsample=60, random_state=0, ax=ax,
                                        ice_lines_kw={"color": "#A78BFA", "alpha": .25}, pd_line_kw={"color": "#FF6B1A", "lw": 3})
fig.suptitle("Partial dependence (orange) and ICE curves (purple): how each feature moves predicted price")
fig.tight_layout(); fig.savefig("session-15_partial_dependence.png", dpi=100); plt.show()
