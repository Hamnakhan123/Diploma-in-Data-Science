"""Real-life problem: compare three classifiers for loan default and pick a cost-optimal threshold."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

rng = np.random.default_rng(11)
n = 3000
df = pd.DataFrame({
    "monthly_income_k": rng.lognormal(4.0, 0.45, n).round(1),      # thousand Rs
    "loan_amount_k": rng.uniform(20, 300, n).round(0),
    "existing_loans": rng.poisson(1.0, n),
    "years_employed": rng.exponential(4, n).round(1),
    "late_payments_12m": rng.poisson(0.6, n),
})
z = (-2.2 + 0.012 * df.loan_amount_k - 0.025 * df.monthly_income_k + 0.45 * df.existing_loans
     - 0.15 * df.years_employed + 0.7 * df.late_payments_12m)
df["default"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
print("default rate:", round(df["default"].mean(), 3))

X, y = df.drop(columns="default"), df["default"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
models = {"KNN (k=25)": make_pipeline(StandardScaler(), KNeighborsClassifier(25)),
          "Gaussian NB": GaussianNB(),
          "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression())}
print(f"\n{'model':<22}{'accuracy':>9}{'ROC AUC':>9}")
for name, m in models.items():
    m.fit(X_tr, y_tr)
    p = m.predict_proba(X_te)[:, 1]
    print(f"{name:<22}{accuracy_score(y_te, p >= .5):>9.3f}{roc_auc_score(y_te, p):>9.3f}")

# Pick the threshold with the lowest business cost
lr = models["Logistic regression"]; p = lr.predict_proba(X_te)[:, 1]
COST_FN, COST_FP = 80_000, 2_000
print("\nthreshold  missed defaulters  extra reviews   total cost (Rs)")
best = None
for t in [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01]:
    tn, fp, fn, tp = confusion_matrix(y_te, p >= t).ravel()
    cost = fn * COST_FN + fp * COST_FP
    best = min(best or (cost, t), (cost, t))
    print(f"   {t:<5}        {fn:>4}             {fp:>4}       {cost:>12,}")
print(f"\nbest threshold: {best[1]}  (cost Rs {best[0]:,})")
print(f"theory: flag when p > C_FP / (C_FP + C_FN) = {COST_FP / (COST_FP + COST_FN):.3f}")

coefs = pd.Series(lr[-1].coef_[0], index=X.columns).sort_values()
print("\nstandardised coefficients (positive = raises default risk):\n", coefs.round(2).to_string())
