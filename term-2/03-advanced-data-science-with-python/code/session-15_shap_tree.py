"""SHAP values for a gradient-boosting classifier: additivity check, one explained prediction, global ranking, beeswarm."""
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
warnings.filterwarnings("ignore")

rng = np.random.default_rng(1)
n = 3000
X = pd.DataFrame({"income_k": rng.lognormal(4.2, .4, n).round(), "credit_score": rng.normal(650, 70, n).round(),
                  "debt_ratio": rng.beta(2, 5, n).round(3), "late_payments": rng.poisson(.7, n),
                  "years_employed": rng.exponential(5, n).round(1)})
z = -1 + .02 * (X.income_k - 70) + .015 * (X.credit_score - 650) - 6 * (X.debt_ratio - .3) - .8 * X.late_payments + .08 * X.years_employed
y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)                           # 1 = loan approved
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.25, random_state=0)
model = GradientBoostingClassifier(random_state=0).fit(X_tr, y_tr)
print(f"test accuracy: {model.score(X_te, y_te):.3f}")

explainer = shap.TreeExplainer(model)
sv = explainer(X_te)                       # values are in log-odds for this model
base = float(np.ravel(explainer.expected_value)[0])
i = int(np.argmin(model.predict_proba(X_te)[:, 1]))                    # the most strongly rejected applicant
row = sv[i]
logit = model.decision_function(X_te.iloc[[i]])[0]
print(f"\napplicant #{i}: P(approve) = {model.predict_proba(X_te.iloc[[i]])[0, 1]:.3f}")
print(f"base value (log-odds) {base:.3f} + sum of SHAP {row.values.sum():.3f} = {base + row.values.sum():.3f} | model log-odds {logit:.3f}")
for name, val, v in sorted(zip(X.columns, X_te.iloc[i], row.values), key=lambda t: -abs(t[2])):
    print(f"  {name:<15} = {val:>8}   SHAP {v:+.3f}  ({'pushes towards approve' if v > 0 else 'pushes towards reject'})")

mean_abs = pd.Series(np.abs(sv.values).mean(0), index=X.columns).sort_values(ascending=False)
print("\nglobal importance (mean |SHAP|):\n", mean_abs.round(3).to_string())

shap.plots.beeswarm(sv, show=False)
plt.title("SHAP beeswarm: each dot is one applicant"); plt.tight_layout()
plt.savefig("session-15_shap_beeswarm.png", dpi=100, bbox_inches="tight"); plt.show(); plt.close("all")
shap.plots.waterfall(row, show=False)
plt.title(f"Why applicant #{i} was rejected"); plt.tight_layout()
plt.savefig("session-15_shap_waterfall.png", dpi=100, bbox_inches="tight"); plt.show()
