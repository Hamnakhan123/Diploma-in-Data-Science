"""Use a validation curve to see underfitting and overfitting as Ridge alpha changes."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y=True)
pipe = make_pipeline(PolynomialFeatures(2), StandardScaler(), Ridge())   # 65 features incl. interactions
alphas = np.logspace(-3, 4, 15)
tr, va = validation_curve(pipe, X, y, param_name="ridge__alpha", param_range=alphas, cv=5, scoring="r2")
best = alphas[va.mean(1).argmax()]
for a, t, v in zip(alphas[::2], tr.mean(1)[::2], va.mean(1)[::2]):
    print(f"alpha {a:>10.3f}   train R2 {t:.3f}   validation R2 {v:.3f}")
print(f"best alpha = {best:.2f} with validation R2 = {va.mean(1).max():.3f}")

plt.figure(figsize=(7, 4))
plt.semilogx(alphas, tr.mean(1), "o-", color="#FF6B1A", label="training R²")
plt.semilogx(alphas, va.mean(1), "s-", color="#A78BFA", label="validation R²")
plt.axvline(best, ls="--", color="grey"); plt.text(best, .1, " best alpha")
plt.xlabel("alpha"); plt.ylabel("R²"); plt.title("Validation curve: overfit (left) → underfit (right)"); plt.legend()
plt.tight_layout(); plt.savefig("session-03_validation_curve.png", dpi=110); plt.show()
