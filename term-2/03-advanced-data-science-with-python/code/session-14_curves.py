"""Plot ROC and precision-recall curves, and a calibration (reliability) diagram before and after calibration."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, brier_score_loss

X, y = make_classification(n_samples=12000, n_features=15, n_informative=6, n_redundant=6, weights=[.85], random_state=1)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.4, stratify=y, random_state=0)
models = {"Naive Bayes": GaussianNB(), "Gradient boosting": HistGradientBoostingClassifier(random_state=0),
          "Naive Bayes + isotonic calibration": CalibratedClassifierCV(GaussianNB(), method="isotonic", cv=5)}
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
for (name, m), c in zip(models.items(), ["#A78BFA", "#FF6B1A", "#F43F5E"]):
    p = m.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, p); prec, rec, _ = precision_recall_curve(y_te, p)
    frac, mean_p = calibration_curve(y_te, p, n_bins=10, strategy="quantile")
    print(f"{name:<36} ROC AUC {roc_auc_score(y_te, p):.3f}  PR AUC {average_precision_score(y_te, p):.3f}  Brier {brier_score_loss(y_te, p):.4f}")
    ax[0].plot(fpr, tpr, color=c, label=name); ax[1].plot(rec, prec, color=c, label=name)
    ax[2].plot(mean_p, frac, "o-", color=c, label=name)
ax[0].plot([0, 1], [0, 1], "--", color="grey"); ax[0].set_title("ROC curve"); ax[0].set_xlabel("false positive rate"); ax[0].set_ylabel("true positive rate (recall)")
ax[1].axhline(y_te.mean(), ls="--", color="grey"); ax[1].set_title("Precision-recall curve"); ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision")
ax[2].plot([0, 1], [0, 1], "--", color="grey"); ax[2].set_title("Calibration (reliability) diagram"); ax[2].set_xlabel("predicted probability"); ax[2].set_ylabel("observed frequency")
for a in ax: a.legend(fontsize=7)
fig.tight_layout(); fig.savefig("session-14_evaluation_curves.png", dpi=100); plt.show()
