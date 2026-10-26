"""Compute Gini impurity, entropy and the information gain of candidate splits by hand."""
import numpy as np

def gini(labels):
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return 1 - np.sum(p ** 2)

def entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return -np.sum(p * np.log2(p))

def gain(parent, left, right, measure=gini):
    n = len(parent)
    return measure(parent) - (len(left) / n * measure(left) + len(right) / n * measure(right))

# 10 customers: did they buy the premium plan? (1 = yes)
age     = np.array([22, 25, 28, 35, 38, 42, 47, 51, 55, 60])
salary  = np.array([30, 45, 32, 80, 95, 40, 120, 85, 38, 110])   # thousand Rs
bought  = np.array([0, 0, 0, 1, 1, 0, 1, 1, 0, 1])

print(f"parent: Gini = {gini(bought):.3f}, entropy = {entropy(bought):.3f}")
for name, col in [("age", age), ("salary", salary)]:
    best = max(((gain(bought, bought[col <= t], bought[col > t]), t) for t in np.unique(col)[:-1]))
    print(f"best split on {name:<6}: <= {best[1]:>3}  gain (Gini) = {best[0]:.3f}")
t = 45
print(f"\nsalary <= {t}: left {bought[salary <= t]}, right {bought[salary > t]}")
print(f"info gain (entropy) = {gain(bought, bought[salary <= t], bought[salary > t], entropy):.3f}")
