"""Real-life problem: telecom churn with tuned tree ensembles, lift in the top decile and feature importance."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(21)
n = 6000
df = pd.DataFrame({
    "tenure_months": rng.integers(1, 72, n),
    "monthly_spend": rng.gamma(3, 300, n).round(),
    "data_gb": rng.gamma(2, 4, n).round(1),
    "complaints_90d": rng.poisson(0.4, n),
    "dropped_calls_pct": rng.beta(2, 30, n).round(3) * 100,
    "on_contract": (rng.random(n) < .35).astype(int),
    "recharge_gap_days": rng.exponential(6, n).round(),
})
z = (-1.1 - 0.035 * df.tenure_months + 0.9 * df.complaints_90d + 0.12 * df.dropped_calls_pct
     - 1.4 * df.on_contract + 0.09 * df.recharge_gap_days - 0.04 * df.data_gb
     + 0.8 * ((df.monthly_spend > 1200) & (df.data_gb < 3)))                # paying a lot, using little
df["churn"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
print("churn rate:", round(df.churn.mean(), 3))

X, y = df.drop(columns="churn"), df["churn"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.3, random_state=0, stratify=y)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
searches = {
    "Decision tree": RandomizedSearchCV(DecisionTreeClassifier(random_state=0),
        {"max_depth": [3, 4, 5, 6, 8, 10], "min_samples_leaf": [5, 10, 20, 50, 100]}, n_iter=15, cv=cv, scoring="roc_auc", random_state=0),
    "Random forest": RandomizedSearchCV(RandomForestClassifier(random_state=0, n_jobs=-1),
        {"n_estimators": [200, 400], "max_depth": [None, 8, 12], "min_samples_leaf": [1, 5, 10], "max_features": ["sqrt", 0.5]},
        n_iter=8, cv=cv, scoring="roc_auc", random_state=0),
    "Gradient boosting": RandomizedSearchCV(HistGradientBoostingClassifier(random_state=0),
        {"learning_rate": [0.03, 0.06, 0.1], "max_depth": [3, 5, None], "max_iter": [200, 400], "l2_regularization": [0, 1]},
        n_iter=8, cv=cv, scoring="roc_auc", random_state=0),
}
base_rate = y_te.mean()
print(f"\n{'model':<18}{'CV AUC':>7}{'test AUC':>9}{'churners in top 10%':>21}{'lift':>6}")
for name, s in searches.items():
    s.fit(X_tr, y_tr)
    p = s.predict_proba(X_te)[:, 1]
    top = np.argsort(-p)[: len(p) // 10]
    hit = y_te.values[top].mean()
    print(f"{name:<18}{s.best_score_:>7.3f}{roc_auc_score(y_te, p):>9.3f}{hit:>20.1%}{hit / base_rate:>6.1f}")

best = searches["Gradient boosting"].best_estimator_
imp = permutation_importance(best, X_te, y_te, scoring="roc_auc", n_repeats=5, random_state=0)
order = imp.importances_mean.argsort()
plt.figure(figsize=(7, 4))
plt.barh(X.columns[order], imp.importances_mean[order], color="#FF6B1A")
plt.xlabel("drop in ROC AUC when shuffled"); plt.title("What drives churn? (permutation importance)")
plt.tight_layout(); plt.savefig("session-04_churn_importance.png", dpi=110); plt.show()
print("\ntop drivers:", list(X.columns[order[::-1][:3]]))
