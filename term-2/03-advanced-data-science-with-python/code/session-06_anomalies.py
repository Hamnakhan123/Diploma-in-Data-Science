"""Detect injected anomalies in sensor data with z-score, IQR, Isolation Forest, LOF and One-Class SVM."""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score

rng = np.random.default_rng(3)
# Factory motor: temperature (°C) and vibration (mm/s) are correlated in normal operation
temp = rng.normal(70, 4, 980)
vib = 2 + 0.05 * (temp - 70) + rng.normal(0, .15, 980)
normal = np.c_[temp, vib]
# 20 anomalies: normal temperature but unusual vibration for that temperature (a bearing fault)
t_bad = rng.normal(70, 4, 20)
bad = np.c_[t_bad, 2 + 0.05 * (t_bad - 70) + rng.choice([-1, 1], 20) * rng.uniform(.6, 1.0, 20)]
X = np.vstack([normal, bad]); y = np.r_[np.zeros(980), np.ones(20)].astype(int)
Xs = StandardScaler().fit_transform(X)

def report(name, flags):
    print(f"{name:<22} flagged {flags.sum():>3}  precision {precision_score(y, flags):.2f}  recall {recall_score(y, flags):.2f}")

z = np.abs(Xs).max(axis=1); report("z-score > 3 (per col)", (z > 3).astype(int))
q1, q3 = np.percentile(X, [25, 75], axis=0); iqr = q3 - q1
report("IQR rule (per col)", ((X < q1 - 1.5 * iqr) | (X > q3 + 1.5 * iqr)).any(axis=1).astype(int))
report("Isolation Forest", (IsolationForest(contamination=.02, random_state=0).fit_predict(Xs) == -1).astype(int))
report("Local Outlier Factor", (LocalOutlierFactor(n_neighbors=30, contamination=.02).fit_predict(Xs) == -1).astype(int))
report("One-Class SVM", (OneClassSVM(nu=.02, gamma=.5).fit(Xs).predict(Xs) == -1).astype(int))
