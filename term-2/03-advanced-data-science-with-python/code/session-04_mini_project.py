"""Mini project: tune three classifiers for loan approval, compare on a test set, export decision rules."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

rng = np.random.default_rng(8)
n = 2000
X = pd.DataFrame({"income_k": rng.lognormal(4.2, .4, n), "credit_score": rng.normal(650, 70, n).clip(300, 850),
                  "debt_ratio": rng.beta(2, 5, n), "years_job": rng.exponential(5, n)})
z = -2.4 + 0.04 * X.income_k + 0.024 * (X.credit_score - 600) - 10 * X.debt_ratio + 0.2 * X.years_job
y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)                # 1 = approve (about half)
print("approval rate:", round(y.mean(), 3))
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.25, random_state=0, stratify=y)
cv = StratifiedKFold(5, shuffle=True, random_state=0)

tuned = {
    "tree": GridSearchCV(DecisionTreeClassifier(random_state=0), {"max_depth": [2, 3, 4, 6], "min_samples_leaf": [10, 30]}, cv=cv),
    "forest": GridSearchCV(RandomForestClassifier(random_state=0, n_jobs=-1), {"n_estimators": [200], "min_samples_leaf": [1, 5, 15]}, cv=cv),
    "svm": GridSearchCV(make_pipeline(StandardScaler(), SVC()), {"svc__C": [0.3, 1, 3, 10], "svc__gamma": ["scale", 0.05]}, cv=cv),
}
for name, g in tuned.items():
    g.fit(X_tr, y_tr)
    print(f"{name:<7} best CV acc {g.best_score_:.3f} | test acc {g.score(X_te, y_te):.3f} | {g.best_params_}")
winner = max(tuned, key=lambda k: tuned[k].best_score_)
print(f"\nwinner by CV: {winner}\n")
print(classification_report(y_te, tuned[winner].predict(X_te), target_names=["decline", "approve"]))
# The tuned tree is deep; for a policy document fit a short depth-2 tree and accept a little less accuracy
policy = DecisionTreeClassifier(max_depth=2, min_samples_leaf=30, random_state=0).fit(X_tr, y_tr)
print(f"Readable depth-2 policy tree (test acc {policy.score(X_te, y_te):.3f}):")
print(export_text(policy, feature_names=list(X.columns), decimals=2))
