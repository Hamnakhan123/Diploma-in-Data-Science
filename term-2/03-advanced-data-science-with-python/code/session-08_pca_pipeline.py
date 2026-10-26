"""PCA as preprocessing: accuracy and training time vs number of components; tune n_components with CV."""
import time
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

X, y = load_digits(return_X_y=True)
print("components  CV accuracy  time (s)")
for k in [5, 10, 20, 30, 64]:
    pipe = make_pipeline(StandardScaler(), PCA(k, random_state=0), LogisticRegression(max_iter=3000))
    t0 = time.perf_counter(); acc = cross_val_score(pipe, X, y, cv=5).mean(); dt = time.perf_counter() - t0
    print(f"{k:>10}  {acc:>11.3f}  {dt:>8.2f}")

pipe = Pipeline([("scale", StandardScaler()), ("pca", PCA(random_state=0)), ("svm", SVC())])
gs = GridSearchCV(pipe, {"pca__n_components": [10, 20, 30, 0.95], "svm__C": [1, 10]}, cv=5).fit(X, y)
print(f"\nbest: {gs.best_params_}  CV accuracy {gs.best_score_:.3f}")
print("note: n_components=0.95 means 'keep enough components for 95% of variance' ->",
      gs.best_estimator_.named_steps["pca"].n_components_ if gs.best_params_["pca__n_components"] == 0.95 else "not chosen")
