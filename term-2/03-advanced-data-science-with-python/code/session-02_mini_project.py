"""Mini project: predict whether students pass, choose a model by CV, explain predictions."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

rng = np.random.default_rng(5)
n = 600
students = pd.DataFrame({"study_hours": rng.uniform(0, 15, n), "attendance": rng.uniform(40, 100, n),
                         "assignments_done": rng.integers(0, 11, n), "past_gpa": rng.uniform(1.5, 4.0, n)})
z = -9 + 0.35 * students.study_hours + 0.05 * students.attendance + 0.25 * students.assignments_done + 0.9 * students.past_gpa
students["passed"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)

X, y = students.drop(columns="passed"), students["passed"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
candidates = {"logistic": make_pipeline(StandardScaler(), LogisticRegression()),
              "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(15))}
scores = {k: cross_val_score(m, X_tr, y_tr, cv=5).mean() for k, m in candidates.items()}
print("CV accuracy:", {k: round(float(v), 3) for k, v in scores.items()})
best_name = max(scores, key=scores.get)
model = candidates[best_name].fit(X_tr, y_tr)
print(f"chosen model: {best_name}\n")
print(classification_report(y_te, model.predict(X_te), target_names=["fail", "pass"]))

new = pd.DataFrame({"study_hours": [2, 8, 12], "attendance": [55, 80, 95],
                    "assignments_done": [3, 7, 10], "past_gpa": [2.0, 2.9, 3.6]})
coef = model[-1].coef_[0] if best_name == "logistic" else None
for i, row in new.iterrows():
    p = model.predict_proba(row.to_frame().T)[0, 1]
    msg = f"student {i + 1}: P(pass) = {p:.2f}"
    if coef is not None:
        contrib = model[0].transform(row.to_frame().T)[0] * coef
        msg += f" | biggest factor: {X.columns[np.argmax(np.abs(contrib))]}"
    print(msg)
