"""Compare a Python loop with NumPy vectorisation and show broadcasting."""
import time
import numpy as np

rng = np.random.default_rng(42)                     # reproducible random numbers
X = rng.normal(loc=[50, 1000, 3], scale=[10, 200, 1], size=(200_000, 3))
print("shape of X:", X.shape)

# --- 1. Pure Python loop (slow) ---------------------------------------
t0 = time.perf_counter()
means = [sum(X[:, j]) / len(X) for j in range(3)]
stds = [(sum((x - means[j]) ** 2 for x in X[:, j]) / len(X)) ** 0.5 for j in range(3)]
Z_loop = [[(X[i, j] - means[j]) / stds[j] for j in range(3)] for i in range(len(X))]
t_loop = time.perf_counter() - t0

# --- 2. NumPy broadcasting (fast) -------------------------------------
t0 = time.perf_counter()
Z = (X - X.mean(axis=0)) / X.std(axis=0)            # (200000,3) - (3,) -> broadcast
t_vec = time.perf_counter() - t0

print(f"loop time       : {t_loop:.2f} s")
print(f"vectorised time : {t_vec:.4f} s")
print(f"speed-up        : about {t_loop / t_vec:,.0f}x")
print("same answer?    :", np.allclose(Z, np.array(Z_loop)))
print("means after     :", Z.mean(axis=0).round(6))
print("stds after      :", Z.std(axis=0).round(6))

# Boolean masks: select rows without a loop
mask = (X[:, 0] > 60) & (X[:, 2] < 2)
print("rows with feature0 > 60 and feature2 < 2:", mask.sum())
