"""Real-life problem: pharmacy inventory as an MDP solved by value iteration, checked by simulation and Q-learning."""
import numpy as np
import matplotlib.pyplot as plt

CAP, MAX_ORDER, gamma = 10, 5, 0.95
PROFIT, HOLD, ORDER_FEE, LOST = 120, 8, 60, 100
demand_p = np.array([.05, .10, .20, .25, .20, .12, .08])          # P(demand = 0..6) per day
demand_vals = np.arange(len(demand_p))
print(f"expected daily demand: {demand_p @ demand_vals:.2f} boxes")

def outcome(stock, order, d):
    avail = min(CAP, stock + order)
    sold = min(avail, d)
    nxt = avail - sold
    reward = PROFIT * sold - HOLD * nxt - (ORDER_FEE if order > 0 else 0) - LOST * (d - sold)
    return nxt, reward

def q_values(V, s):
    return np.array([sum(p * (r + gamma * V[n]) for p, (n, r) in zip(demand_p, (outcome(s, a, d) for d in demand_vals)))
                     for a in range(MAX_ORDER + 1)])

V = np.zeros(CAP + 1)
for sweep in range(1000):
    V_new = np.array([q_values(V, s).max() for s in range(CAP + 1)])
    if np.abs(V_new - V).max() < 1e-4: break
    V = V_new
policy = np.array([q_values(V, s).argmax() for s in range(CAP + 1)])
print(f"value iteration converged in {sweep + 1} sweeps")
print("stock at night :", list(range(CAP + 1)))
print("optimal order  :", policy.tolist())
print("order-up-to level implied:", sorted(set((np.arange(CAP + 1) + policy)[policy > 0].tolist())))

def simulate(rule, days=20000, seed=1):
    rng = np.random.default_rng(seed); s, total = 5, 0.0
    for d in rng.choice(demand_vals, days, p=demand_p):
        s, r = outcome(s, rule(s), d); total += r
    return total / days

rules = {"optimal MDP policy": lambda s: policy[s], "never order": lambda s: 0,
         "always order 3": lambda s: 3, "order 5 when stock < 3": lambda s: 5 if s < 3 else 0}
print("\naverage daily profit by simulation:")
for name, rule in rules.items():
    print(f"  {name:<24} Rs {simulate(rule):7.1f}")

# Model-free check: Q-learning from simulated days only
rng = np.random.default_rng(3); Q = np.zeros((CAP + 1, MAX_ORDER + 1)); s = 5
for t in range(300_000):
    eps = max(.02, 1 - t / 150_000)
    a = rng.integers(MAX_ORDER + 1) if rng.random() < eps else int(Q[s].argmax())
    s2, r = outcome(s, a, rng.choice(demand_vals, p=demand_p))
    Q[s, a] += 0.05 * (r + gamma * Q[s2].max() - Q[s, a]); s = s2
print("\nQ-learning policy:", Q.argmax(axis=1).tolist())
print(f"Q-learning daily profit: Rs {simulate(lambda s: int(Q[s].argmax())):.1f}")

plt.figure(figsize=(7, 3.8))
plt.bar(range(CAP + 1), policy, color="#FF6B1A", label="value iteration")
plt.plot(range(CAP + 1), Q.argmax(axis=1), "o--", color="#A78BFA", label="Q-learning")
plt.xlabel("boxes on shelf at night"); plt.ylabel("boxes to order"); plt.title("Optimal pharmacy ordering rule")
plt.legend(); plt.tight_layout(); plt.savefig("session-11_pharmacy_policy.png", dpi=110); plt.show()
