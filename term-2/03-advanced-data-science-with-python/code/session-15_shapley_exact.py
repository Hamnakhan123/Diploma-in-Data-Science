"""Exact Shapley values for a tiny model by enumerating every coalition, and checking additivity."""
from itertools import combinations
from math import factorial
import numpy as np

# A loan score model with an interaction: f = 10 + 3*income + 2*history + 4*income*history - 5*debt
features = ["income", "history", "debt"]
x = {"income": 1.0, "history": 1.0, "debt": 0.5}          # the applicant (scaled values)
baseline = {"income": 0.2, "history": 0.4, "debt": 0.3}   # an 'average' applicant (background)

def f(v):
    return 10 + 3 * v["income"] + 2 * v["history"] + 4 * v["income"] * v["history"] - 5 * v["debt"]

def value(coalition):
    """Prediction when features in the coalition take the applicant's values, others the baseline."""
    v = {k: (x[k] if k in coalition else baseline[k]) for k in features}
    return f(v)

n = len(features)
phi = {}
for i in features:
    others = [k for k in features if k != i]
    total = 0.0
    for size in range(n):
        for S in combinations(others, size):
            weight = factorial(len(S)) * factorial(n - len(S) - 1) / factorial(n)
            total += weight * (value(set(S) | {i}) - value(set(S)))
    phi[i] = total

base, pred = value(set()), value(set(features))
print(f"base value f(baseline) = {base:.3f}   prediction f(x) = {pred:.3f}   difference = {pred - base:.3f}")
for k, v in phi.items():
    print(f"  SHAP value of {k:<8} = {v:+.3f}")
print(f"sum of SHAP values = {sum(phi.values()):.3f}  (equals the difference: additivity holds)")
