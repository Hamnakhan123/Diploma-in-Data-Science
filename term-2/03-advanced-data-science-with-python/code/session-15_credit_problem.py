"""Real-life problem: SHAP reason codes for rejected loan applicants, a fairness check, and ONNX export."""
import warnings
import numpy as np
import pandas as pd
import shap
import onnxruntime as ort
from skl2onnx import to_onnx
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
warnings.filterwarnings("ignore")

rng = np.random.default_rng(8)
n = 5000
df = pd.DataFrame({"income_k": rng.lognormal(4.2, .45, n).round(), "credit_score": rng.normal(640, 75, n).round(),
                   "debt_ratio": rng.beta(2, 5, n).round(3), "late_payments": rng.poisson(.8, n),
                   "years_employed": rng.exponential(5, n).round(1), "existing_loans": rng.poisson(1, n),
                   "gender_female": (rng.random(n) < .4).astype(int)})
# Historical decisions contain a small unfair penalty for women (bias in past data)
z = (-0.8 + .02 * (df.income_k - 70) + .014 * (df.credit_score - 640) - 5 * (df.debt_ratio - .3)
     - .7 * df.late_payments + .07 * df.years_employed - .3 * df.existing_loans - .5 * df.gender_female)
df["approved"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
X, y = df.drop(columns="approved"), df["approved"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.3, random_state=0, stratify=y)

model = GradientBoostingClassifier(random_state=0).fit(X_tr, y_tr)
sv = shap.TreeExplainer(model)(X_te)
imp = pd.Series(np.abs(sv.values).mean(0), index=X.columns).sort_values(ascending=False)
print(f"model with gender: test accuracy {model.score(X_te, y_te):.3f}")
print("global importance (mean |SHAP|):", imp.round(3).to_dict())

pred = model.predict(X_te)
rates = pd.Series(pred).groupby(X_te["gender_female"].values).mean()
print(f"\nFAIRNESS CHECK -> approval rate men {rates[0]:.1%} vs women {rates[1]:.1%} "
      f"| mean SHAP of gender for women: {sv.values[X_te.gender_female.values == 1, list(X.columns).index('gender_female')].mean():+.3f}")

# Remove the sensitive attribute and retrain
fair_cols = [c for c in X.columns if c != "gender_female"]
fair = GradientBoostingClassifier(random_state=0).fit(X_tr[fair_cols], y_tr)
pred_f = fair.predict(X_te[fair_cols])
rates_f = pd.Series(pred_f).groupby(X_te["gender_female"].values).mean()
print(f"model without gender: accuracy {fair.score(X_te[fair_cols], y_te):.3f} | approval men {rates_f[0]:.1%} vs women {rates_f[1]:.1%}")

# Reason codes for rejected applicants (from the fair model)
phrases = {"income_k": "income is low for the amount requested", "credit_score": "credit score is below our range",
           "debt_ratio": "existing debt is high compared with income", "late_payments": "recent late payments on record",
           "years_employed": "short employment history", "existing_loans": "several loans already active"}
sv_f = shap.TreeExplainer(fair)(X_te[fair_cols])
rejected = np.where(pred_f == 0)[0][:4]
print("\nREASON CODES (top 3 negative SHAP contributions):")
for r in rejected:
    contrib = pd.Series(sv_f.values[r], index=fair_cols).sort_values()
    reasons = [phrases[c] for c in contrib.index[:3] if contrib[c] < 0]
    print(f"  applicant {X_te.index[r]}: " + "; ".join(reasons))

onx = to_onnx(fair, X_tr[fair_cols].values[:1].astype(np.float32), options={id(fair): {"zipmap": False}}, target_opset=17)
sess = ort.InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
onnx_pred = sess.run(None, {sess.get_inputs()[0].name: X_te[fair_cols].values.astype(np.float32)})[0]
print(f"\nONNX export of the fair model agrees with scikit-learn on {np.mean(onnx_pred == pred_f):.1%} of test applicants")
