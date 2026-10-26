"""Mini project: a function that produces a complete evaluation report for any fitted binary classifier."""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, brier_score_loss, log_loss)

def evaluation_report(name, model, X_te, y_te, threshold=0.5, positive="malignant"):
    p = model.predict_proba(X_te)[:, 1]; pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    lines = [f"=== {name} (threshold {threshold}) ===",
             f"confusion matrix: TP={tp} FP={fp} FN={fn} TN={tn}",
             f"precision {precision_score(y_te, pred):.3f} | recall {recall_score(y_te, pred):.3f} | F1 {f1_score(y_te, pred):.3f}",
             f"ROC AUC {roc_auc_score(y_te, p):.3f} | PR AUC {average_precision_score(y_te, p):.3f}",
             f"log loss {log_loss(y_te, p):.3f} | Brier {brier_score_loss(y_te, p):.4f}",
             f"plain-language: of {tp + fn} {positive} cases, the model caught {tp} and missed {fn}; it raised {fp} false alarms."]
    return "\n".join(lines)

X, y = load_breast_cancer(return_X_y=True)
y = 1 - y                                         # make 1 = malignant (the class we must not miss)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.3, stratify=y, random_state=0)
models = {"logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
          "random forest": RandomForestClassifier(300, random_state=0, n_jobs=-1)}
for name, m in models.items():
    m.fit(X_tr, y_tr)
    print(evaluation_report(name, m, X_te, y_te))
    print(evaluation_report(name, m, X_te, y_te, threshold=0.2).split("\n", 1)[1].split("\n")[-1], "(threshold 0.2)\n")
