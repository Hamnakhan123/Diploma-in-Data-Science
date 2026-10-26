"""Logistic regression: sigmoid, from-scratch gradient descent, scikit-learn, odds ratios, thresholds."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

rng = np.random.default_rng(3)
# Students: hours studied -> passed (1) or failed (0)
hours = rng.uniform(0, 10, 200)
p_true = 1 / (1 + np.exp(-(1.2 * hours - 6)))
passed = (rng.random(200) < p_true).astype(int)

sigmoid = lambda z: 1 / (1 + np.exp(-z))

# From scratch: gradient descent on log-loss
w = b = 0.0
for _ in range(40000):                              # many small steps: hours are not scaled
    p = sigmoid(w * hours + b)
    w -= 0.1 * np.mean((p - passed) * hours)
    b -= 0.1 * np.mean(p - passed)
loss = -np.mean(passed * np.log(p) + (1 - passed) * np.log(1 - p))
print(f"from scratch : w = {w:.2f}, b = {b:.2f}, log-loss = {loss:.3f}")

clf = LogisticRegression(C=1e6).fit(hours.reshape(-1, 1), passed)      # huge C = almost no regularisation
print(f"scikit-learn : w = {clf.coef_[0][0]:.2f}, b = {clf.intercept_[0]:.2f}")
print(f"odds ratio per extra hour: {np.exp(clf.coef_[0][0]):.2f}x")
print(f"50% pass point: {-clf.intercept_[0] / clf.coef_[0][0]:.1f} hours")
for h in [2, 5, 8]:
    print(f"P(pass | {h} h) = {clf.predict_proba([[h]])[0, 1]:.3f}")

probs = clf.predict_proba(hours.reshape(-1, 1))[:, 1]
print("\nthreshold  TN  FP  FN  TP")
for t in [0.3, 0.5, 0.7]:
    tn, fp, fn, tp = confusion_matrix(passed, (probs >= t).astype(int)).ravel()
    print(f"   {t:.1f}    {tn:>3} {fp:>3} {fn:>3} {tp:>3}")

grid = np.linspace(0, 10, 200)
plt.figure(figsize=(7, 4))
plt.scatter(hours, passed + rng.normal(0, .02, 200), s=12, alpha=.6, color="#A78BFA", label="students")
plt.plot(grid, clf.predict_proba(grid.reshape(-1, 1))[:, 1], color="#FF6B1A", lw=2.5, label="P(pass)")
plt.axhline(0.5, ls="--", color="grey"); plt.xlabel("hours studied"); plt.ylabel("probability of passing")
plt.title("Logistic regression: the S-shaped sigmoid curve"); plt.legend()
plt.tight_layout(); plt.savefig("session-02_logistic_sigmoid.png", dpi=110); plt.show()
