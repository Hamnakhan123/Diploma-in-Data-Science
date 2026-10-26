"""Leakage demo: several rows per patient. Plain KFold over-estimates accuracy; GroupKFold gives the honest number."""
import numpy as np
from sklearn.model_selection import KFold, GroupKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

rng = np.random.default_rng(0)
patients, visits = 200, 6
pid = np.repeat(np.arange(patients), visits)
signature = rng.normal(0, 1, (patients, 8))[pid]                 # each patient has a personal 'fingerprint'
X = signature + rng.normal(0, .3, (len(pid), 8))
risk = rng.random(patients) < .5
y = risk[pid].astype(int)
X[:, 0] += 0.4 * y + rng.normal(0, 1, len(y))                    # only a weak real signal

rf = RandomForestClassifier(200, random_state=0, n_jobs=-1)
plain = cross_val_score(rf, X, y, cv=KFold(5, shuffle=True, random_state=0)).mean()
grouped = cross_val_score(rf, X, y, cv=GroupKFold(5), groups=pid).mean()
print(f"KFold (visits of the same patient in train and test): {plain:.3f}  <- optimistic, memorises patients")
print(f"GroupKFold (each patient only in one fold)          : {grouped:.3f}  <- what you get for NEW patients")
