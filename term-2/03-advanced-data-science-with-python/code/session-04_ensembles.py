"""Single tree vs bagging, random forest, AdaBoost, gradient boosting, voting and stacking."""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier, AdaBoostClassifier,
                              HistGradientBoostingClassifier, VotingClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=1500, n_features=20, n_informative=8, n_redundant=4,
                           flip_y=0.05, class_sep=0.8, random_state=0)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
svm = make_pipeline(StandardScaler(), SVC(probability=True, random_state=0))
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
rf = RandomForestClassifier(300, random_state=0, n_jobs=-1)
models = {
    "Single decision tree": DecisionTreeClassifier(random_state=0),
    "Bagging (100 trees)": BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, random_state=0, n_jobs=-1),
    "Random forest": rf,
    "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=0),
    "Gradient boosting (hist)": HistGradientBoostingClassifier(random_state=0),
    "Soft voting (LR+SVM+RF)": VotingClassifier([("lr", lr), ("svm", svm), ("rf", rf)], voting="soft", n_jobs=-1),
    "Stacking (meta = LR)": StackingClassifier([("svm", svm), ("rf", rf)], final_estimator=LogisticRegression(), n_jobs=-1),
}
print(f"{'model':<26}{'CV accuracy':>12}")
for name, m in models.items():
    s = cross_val_score(m, X, y, cv=cv)
    print(f"{name:<26}{s.mean():>8.3f} ± {s.std():.3f}")

oob = RandomForestClassifier(300, oob_score=True, random_state=0, n_jobs=-1).fit(X, y)
print(f"\nrandom forest out-of-bag score: {oob.oob_score_:.3f} (free validation, no extra split)")
