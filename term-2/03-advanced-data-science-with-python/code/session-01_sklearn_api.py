"""The scikit-learn estimator API: transformers, models, pipelines, cross-validation."""
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

X, y = load_wine(return_X_y=True)            # 178 wines, 13 chemical features, 3 cultivars
print("X shape:", X.shape, "| classes:", sorted(set(y.tolist())))
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# A transformer learns (fit) then changes data (transform)
scaler = StandardScaler().fit(X_tr)
print("first 3 learned means:", scaler.mean_[:3].round(2))

# Swap models with one line: same fit / score calls
models = {
    "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "KNN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
    "KNN without scaling": KNeighborsClassifier(5),
    "Decision tree": DecisionTreeClassifier(random_state=0),
}
print(f"\n{'model':<22}{'test acc':>9}{'5-fold CV':>12}")
for name, model in models.items():
    model.fit(X_tr, y_tr)
    cv = cross_val_score(model, X, y, cv=5)
    print(f"{name:<22}{model.score(X_te, y_te):>9.3f}{cv.mean():>8.3f} ± {cv.std():.3f}")

pipe = models["Logistic regression"]
print("\nprobabilities for first test wine:", pipe.predict_proba(X_te[:1]).round(3))
