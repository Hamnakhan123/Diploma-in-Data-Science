"""Accuracy trap, confusion matrix, precision/recall/F1, ROC AUC vs PR AUC, log loss and MCC on 3% fraud."""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, log_loss, matthews_corrcoef, confusion_matrix)

X, y = make_classification(n_samples=20000, n_features=12, n_informative=6, weights=[.97], flip_y=.01, random_state=4)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.3, stratify=y, random_state=0)
print(f"fraud rate in test set: {y_te.mean():.3%}")

models = {"always 'not fraud'": DummyClassifier(strategy="most_frequent"),
          "logistic regression": LogisticRegression(max_iter=1000),
          "gradient boosting": HistGradientBoostingClassifier(random_state=0)}
print(f"\n{'model':<21}{'acc':>6}{'prec':>6}{'rec':>6}{'F1':>6}{'ROC':>6}{'PR AUC':>8}{'logloss':>8}{'MCC':>6}")
for name, m in models.items():
    m.fit(X_tr, y_tr)
    p = m.predict_proba(X_te)[:, 1]; pred = (p >= .5).astype(int)
    print(f"{name:<21}{accuracy_score(y_te, pred):>6.3f}{precision_score(y_te, pred, zero_division=0):>6.2f}"
          f"{recall_score(y_te, pred):>6.2f}{f1_score(y_te, pred):>6.2f}{roc_auc_score(y_te, p):>6.2f}"
          f"{average_precision_score(y_te, p):>8.2f}{log_loss(y_te, p):>8.3f}{matthews_corrcoef(y_te, pred):>6.2f}")

gb = models["gradient boosting"]; p = gb.predict_proba(X_te)[:, 1]
tn, fp, fn, tp = confusion_matrix(y_te, p >= .5).ravel()
print(f"\nconfusion matrix at 0.5 -> TP {tp}, FN {fn}, FP {fp}, TN {tn}")
for t in [.5, .3, .1]:
    pred = p >= t
    print(f"threshold {t:<4}: precision {precision_score(y_te, pred):.2f}  recall {recall_score(y_te, pred):.2f}  F1 {f1_score(y_te, pred):.2f}")
