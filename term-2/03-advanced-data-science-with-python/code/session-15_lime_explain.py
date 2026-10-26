"""LIME explanation of one prediction from a random forest, its stability across runs, and agreement with SHAP."""
import warnings
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")

rng = np.random.default_rng(1)
n = 3000
X = pd.DataFrame({"income_k": rng.lognormal(4.2, .4, n).round(), "credit_score": rng.normal(650, 70, n).round(),
                  "debt_ratio": rng.beta(2, 5, n).round(3), "late_payments": rng.poisson(.7, n),
                  "years_employed": rng.exponential(5, n).round(1)})
z = -1 + .02 * (X.income_k - 70) + .015 * (X.credit_score - 650) - 6 * (X.debt_ratio - .3) - .8 * X.late_payments + .08 * X.years_employed
y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.25, random_state=0)
rf = RandomForestClassifier(300, min_samples_leaf=5, random_state=0, n_jobs=-1).fit(X_tr, y_tr)

applicant = X_te.iloc[0]
print("applicant:", applicant.to_dict())
print(f"random forest P(approve) = {rf.predict_proba(applicant.to_frame().T)[0, 1]:.3f}")

lime_exp = LimeTabularExplainer(X_tr.values, feature_names=list(X.columns), class_names=["reject", "approve"],
                                mode="classification", discretize_continuous=True, random_state=0)
exp = lime_exp.explain_instance(applicant.values, rf.predict_proba, num_features=5, num_samples=3000)
print("\nLIME explanation (weight towards 'approve'):")
for rule, w in exp.as_list():
    print(f"  {rule:<32} {w:+.3f}")
print(f"LIME local model R2 (how well the line fits locally): {exp.score:.2f}")

tops = []
for seed in range(5):
    e = LimeTabularExplainer(X_tr.values, feature_names=list(X.columns), mode="classification",
                             random_state=seed).explain_instance(applicant.values, rf.predict_proba, num_features=2, num_samples=1000)
    tops.append(tuple(r.split(" ")[0] if r.split(" ")[0] in X.columns else r.split(" ")[2] for r, _ in e.as_list()))
print("\ntop-2 features over 5 LIME runs with different seeds:", tops)

sv = shap.TreeExplainer(rf)(X_te.iloc[[0]])
vals = sv.values[0, :, 1] if sv.values.ndim == 3 else sv.values[0]
print("SHAP ranking for the same applicant:", list(X.columns[np.argsort(-np.abs(vals))][:3]))
