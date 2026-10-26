"""10-armed bandit: greedy, epsilon-greedy, UCB and Thompson sampling, averaged over 200 runs."""
import numpy as np
import matplotlib.pyplot as plt

K, T, RUNS = 10, 1000, 200
def run(strategy, seed):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.1, 0.9, K)                     # true click rates of 10 ad designs (unknown)
    Q, N = np.zeros(K), np.zeros(K)
    a_s, b_s = np.ones(K), np.ones(K)                # Beta(1,1) priors for Thompson
    regret, best = np.zeros(T), p.max()
    for t in range(T):
        if strategy == "greedy":
            a = int(np.argmax(Q))
        elif strategy == "eps-greedy 0.1":
            a = rng.integers(K) if rng.random() < .1 else int(np.argmax(Q))
        elif strategy == "UCB (c=1)":
            a = t if t < K else int(np.argmax(Q + np.sqrt(np.log(t + 1) / N)))
        else:                                        # Thompson sampling
            a = int(np.argmax(rng.beta(a_s, b_s)))
        r = float(rng.random() < p[a])
        N[a] += 1; Q[a] += (r - Q[a]) / N[a]
        a_s[a] += r; b_s[a] += 1 - r
        regret[t] = best - p[a]
    return regret.cumsum(), np.argmax(N) == np.argmax(p)

strategies = ["greedy", "eps-greedy 0.1", "UCB (c=1)", "Thompson"]
plt.figure(figsize=(8, 4.5))
print(f"{'strategy':<16}{'regret after 1000':>18}{'found best arm':>16}")
for s in strategies:
    out = [run(s, k) for k in range(RUNS)]
    reg = np.mean([o[0] for o in out], axis=0); found = np.mean([o[1] for o in out])
    print(f"{s:<16}{reg[-1]:>18.1f}{found:>15.0%}")
    plt.plot(reg, label=s, lw=2)
plt.xlabel("time step"); plt.ylabel("cumulative regret"); plt.title("Exploration strategies on a 10-armed bandit (lower is better)")
plt.legend(); plt.tight_layout(); plt.savefig("session-10_bandit_regret.png", dpi=110); plt.show()
