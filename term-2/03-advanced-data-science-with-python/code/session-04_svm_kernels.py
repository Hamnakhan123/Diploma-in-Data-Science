"""Linear vs RBF SVM on ring-shaped data; the effect of C and gamma; a decision-boundary figure."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X, y = make_circles(n_samples=400, noise=0.12, factor=0.45, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.3, random_state=0)

print("kernel   C      gamma  support vectors  train acc  test acc")
configs = [("linear", 1, "scale"), ("poly", 1, "scale"), ("rbf", 1, 0.1), ("rbf", 1, 1), ("rbf", 100, 50)]
fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
xx, yy = np.meshgrid(np.linspace(-1.6, 1.6, 300), np.linspace(-1.6, 1.6, 300))
for i, (k, C, g) in enumerate(configs):
    m = make_pipeline(StandardScaler(), SVC(kernel=k, C=C, gamma=g, degree=2)).fit(X_tr, y_tr)
    nsv = m[-1].n_support_.sum()
    print(f"{k:<7}{C:>4}  {str(g):>7}  {nsv:>15}  {m.score(X_tr, y_tr):>9.3f}  {m.score(X_te, y_te):>8.3f}")
    if i in (0, 2, 3, 4):
        ax = axes[[0, 2, 3, 4].index(i)]
        Z = m.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z > 0, alpha=.25, cmap="coolwarm"); ax.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=["--", "-", "--"], colors="k", linewidths=.8)
        ax.scatter(X_tr[:, 0], X_tr[:, 1], c=y_tr, cmap="coolwarm", s=10)
        ax.set_title(f"{k}, C={C}, gamma={g}"); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("SVM decision boundaries: a line fails, RBF succeeds, a huge gamma overfits")
fig.tight_layout(); fig.savefig("session-04_svm_kernels.png", dpi=110); plt.show()

# Why the kernel works: add the feature r^2 = x1^2 + x2^2 and the rings become linearly separable
r2 = (X ** 2).sum(axis=1).reshape(-1, 1)
lin = SVC(kernel="linear").fit(np.c_[X, r2][:280], y[:280])
print(f"\nlinear SVM after adding x1^2 + x2^2 as a 3rd feature: test acc = {lin.score(np.c_[X, r2][280:], y[280:]):.3f}")
