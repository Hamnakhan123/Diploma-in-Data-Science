"""Hyperparameter tuning strategies compared on the same SVM: grid, random, halving, nested CV."""
import time
import numpy as np
from scipy.stats import loguniform
from sklearn.datasets import load_breast_cancer
from sklearn.experimental import enable_halving_search_cv  # noqa: F401  (activates Halving*)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV,
                                     cross_val_score, train_test_split, StratifiedKFold)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.25, random_state=0, stratify=y)
pipe = make_pipeline(StandardScaler(), SVC())
grid = {"svc__C": np.logspace(-2, 3, 6), "svc__gamma": np.logspace(-4, 1, 6)}
cv = StratifiedKFold(5, shuffle=True, random_state=0)

searches = {
    "Grid search (36 combos)": GridSearchCV(pipe, grid, cv=cv),
    "Random search (12 draws)": RandomizedSearchCV(pipe, {"svc__C": loguniform(1e-2, 1e3), "svc__gamma": loguniform(1e-4, 10)},
                                                   n_iter=12, cv=cv, random_state=0),
    "Successive halving": HalvingGridSearchCV(pipe, grid, cv=cv, factor=3, random_state=0),
}
print(f"{'strategy':<26}{'fits':>6}{'time s':>8}{'best CV':>9}{'test':>7}  best params")
for name, s in searches.items():
    t0 = time.perf_counter(); s.fit(X_tr, y_tr); dt = time.perf_counter() - t0
    fits = len(s.cv_results_["params"]) * cv.get_n_splits()
    bp = {k.split("__")[1]: round(float(v), 4) for k, v in s.best_params_.items()}
    print(f"{name:<26}{fits:>6}{dt:>8.2f}{s.best_score_:>9.3f}{s.score(X_te, y_te):>7.3f}  {bp}")

# Nested CV: an honest estimate of 'grid search + SVM' as a whole procedure
outer = StratifiedKFold(5, shuffle=True, random_state=1)
nested = cross_val_score(GridSearchCV(pipe, grid, cv=3), X, y, cv=outer)
print(f"\nnested CV accuracy: {nested.mean():.3f} ± {nested.std():.3f}")
