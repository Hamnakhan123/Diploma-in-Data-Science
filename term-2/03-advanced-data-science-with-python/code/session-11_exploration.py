"""Exploration strategies: action probabilities under epsilon-greedy and softmax, then a learning comparison."""
import numpy as np
import matplotlib.pyplot as plt

Q = np.array([5.0, 4.5, 1.0, -20.0])                 # estimated values of 4 actions
def eps_greedy_probs(Q, eps):
    p = np.full(len(Q), eps / len(Q)); p[np.argmax(Q)] += 1 - eps; return p
def softmax_probs(Q, tau):
    z = (Q - Q.max()) / tau                          # subtract max for numerical stability
    e = np.exp(z); return e / e.sum()

print("action values:      ", Q)
print("eps-greedy (0.2):   ", eps_greedy_probs(Q, .2).round(3), "<- the -20 action still gets 5%")
for tau in [10, 2, 0.5, 0.1]:
    print(f"softmax (tau={tau:<4}): ", softmax_probs(Q, tau).round(3))

# Learning comparison on a 10-armed bandit with one disastrous arm
rng_master = np.random.default_rng(0)
def run(strategy, T=2000, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.normal(0, 1, 10); means[3] = -10           # one very costly action
    Qh, N, total = np.zeros(10), np.zeros(10), np.zeros(T)
    for t in range(T):
        if strategy == "eps-greedy 0.1":
            a = rng.integers(10) if rng.random() < .1 else int(np.argmax(Qh))
        elif strategy == "decaying eps":
            a = rng.integers(10) if rng.random() < max(.01, 1 / (1 + t / 50)) else int(np.argmax(Qh))
        else:
            tau = max(.05, 1.0 * .998 ** t)                 # temperature decays from 1.0
            a = rng.choice(10, p=softmax_probs(Qh, tau))
        r = rng.normal(means[a], 1)
        N[a] += 1; Qh[a] += (r - Qh[a]) / N[a]; total[t] = r
    return total.cumsum() / np.arange(1, T + 1), N[3]

plt.figure(figsize=(8, 4))
for s in ["eps-greedy 0.1", "decaying eps", "softmax (decaying tau)"]:
    res = [run(s, seed=k) for k in range(100)]
    curve = np.mean([r[0] for r in res], axis=0); bad = np.mean([r[1] for r in res])
    print(f"{s:<23} average reward {curve[-1]:.3f} | pulls of the disastrous arm: {bad:.0f}")
    plt.plot(curve, label=s)
plt.xlabel("step"); plt.ylabel("average reward so far"); plt.title("Exploration strategies with one costly action")
plt.legend(); plt.tight_layout(); plt.savefig("session-11_exploration.png", dpi=110); plt.show()
