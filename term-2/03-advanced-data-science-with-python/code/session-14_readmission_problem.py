"""Real-life problem: readmission risk model tuned on PR AUC, judged by recall@capacity and calibration."""
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

rng = np.random.default_rng(30)
n = 8000
df = pd.DataFrame({"age": rng.integers(18, 95, n), "length_of_stay": rng.gamma(2, 2.5, n).round(),
                   "prior_admissions": rng.poisson(.8, n), "num_medications": rng.poisson(6, n),
                   "chronic_conditions": rng.poisson(1.5, n), "lives_alone": rng.random(n) < .25,
                   "follow_up_booked": rng.random(n) < .6, "abnormal_labs": rng.poisson(1, n)})
z = (-3.4 + .02 * (df.age - 50) + .08 * df.length_of_stay + .55 * df.prior_admissions + .05 * df.num_medications
     + .25 * df.chronic_conditions + .5 * df.lives_alone - .7 * df.follow_up_booked + .2 * df.abnormal_labs
     + .4 * (df.prior_admissions >= 2) * df.lives_alone)
df["readmitted"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
X, y = df.drop(columns="readmitted").astype(float), df["readmitted"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.25, stratify=y, random_state=0)
print(f"readmission rate: {y.mean():.1%} | train {len(X_tr)} | locked test {len(X_te)}")

cv = StratifiedKFold(5, shuffle=True, random_state=0)
searches = {
    "logistic regression": RandomizedSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        {"logisticregression__C": loguniform(1e-3, 1e2)}, n_iter=12, cv=cv, scoring="average_precision", random_state=0),
    "gradient boosting": RandomizedSearchCV(HistGradientBoostingClassifier(random_state=0),
        {"learning_rate": loguniform(.02, .3), "max_depth": [2, 3, 4, None], "max_iter": randint(100, 500),
         "l2_regularization": loguniform(1e-3, 10)}, n_iter=12, cv=cv, scoring="average_precision", random_state=0),
}
CAP = 0.10
def recall_at_capacity(y_true, p, cap=CAP):
    top = np.argsort(-p)[: int(len(p) * cap)]
    return y_true.values[top].sum() / y_true.sum()

print(f"\n{'model':<22}{'CV PR AUC':>10}{'test PR AUC':>12}{'ROC AUC':>9}{'recall@top10%':>15}{'Brier':>8}")
for name, s in searches.items():
    s.fit(X_tr, y_tr); p = s.predict_proba(X_te)[:, 1]
    print(f"{name:<22}{s.best_score_:>10.3f}{average_precision_score(y_te, p):>12.3f}{roc_auc_score(y_te, p):>9.3f}"
          f"{recall_at_capacity(y_te, p):>15.1%}{brier_score_loss(y_te, p):>8.4f}")

best = max(searches, key=lambda k: searches[k].best_score_)
cal = CalibratedClassifierCV(searches[best].best_estimator_, method="isotonic", cv=5).fit(X_tr, y_tr)
p = cal.predict_proba(X_te)[:, 1]
threshold = np.quantile(p, 1 - CAP)
print(f"\nchosen model: {best} (+ isotonic calibration) | Brier {brier_score_loss(y_te, p):.4f}")
print(f"operating threshold for 100 calls per 1,000 discharges: risk >= {threshold:.2f}")
flagged = p >= threshold
print(f"patients flagged: {flagged.mean():.1%} | precision among calls: {y_te[flagged].mean():.1%} "
      f"(base rate {y_te.mean():.1%}) | readmissions caught: {y_te[flagged].sum() / y_te.sum():.1%}")
bins = pd.cut(p, [0, .05, .1, .2, .3, 1])
print("\ncalibration check on the test set (predicted vs observed):")
print(pd.DataFrame({"predicted": p, "observed": y_te.values}).groupby(bins, observed=True).mean().round(3).to_string())
