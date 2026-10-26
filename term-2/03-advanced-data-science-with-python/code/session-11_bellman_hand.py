"""A student's week as a 3-state MDP: evaluate a fixed policy exactly (linear algebra) and iteratively."""
import numpy as np

states = ["Study", "Social media", "Sleep"]
# Under the student's current policy, transition matrix P[s, s'] and expected reward R[s]
P = np.array([[0.5, 0.3, 0.2],
              [0.4, 0.5, 0.1],
              [0.6, 0.2, 0.2]])
R = np.array([+2.0, -1.0, +1.0])
gamma = 0.9

# Exact solution of V = R + gamma P V  ->  (I - gamma P) V = R
V_exact = np.linalg.solve(np.eye(3) - gamma * P, R)
print("exact values:", {st: round(float(v), 2) for st, v in zip(states, V_exact)})

# Iterative policy evaluation: repeat the Bellman expectation update
V = np.zeros(3)
for k in range(1, 200):
    V_new = R + gamma * P @ V
    if k in (1, 2, 3, 10, 50):
        print(f"sweep {k:>3}: {V_new.round(2)}")
    if np.max(np.abs(V_new - V)) < 1e-6:
        print(f"converged after {k} sweeps: {V_new.round(2)}")
        break
    V = V_new

# One Bellman update by hand for 'Study' using the exact values
s = 0
manual = R[s] + gamma * sum(P[s, j] * V_exact[j] for j in range(3))
print(f"\ncheck: V(Study) = 2 + 0.9 x (0.5 x {V_exact[0]:.2f} + 0.3 x {V_exact[1]:.2f} + 0.2 x {V_exact[2]:.2f}) = {manual:.2f}")
