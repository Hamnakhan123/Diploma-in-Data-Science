"""Cliff Walking: SARSA (on-policy) vs Q-learning (off-policy) vs Monte Carlo control, learning curves and paths."""
import numpy as np
import matplotlib.pyplot as plt

H, W = 4, 12
START, GOAL = (3, 0), (3, 11)
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]           # up, right, down, left
ARROWS = "^>v<"

def env_step(s, a):
    r, c = min(max(s[0] + MOVES[a][0], 0), H - 1), min(max(s[1] + MOVES[a][1], 0), W - 1)
    if r == 3 and 0 < c < 11:
        return START, -100, False                     # fell off the cliff
    return (r, c), -1, (r, c) == GOAL

def policy(Q, s, eps, rng):
    return rng.integers(4) if rng.random() < eps else int(np.argmax(Q[s]))

def train(method, episodes=500, alpha=.5, gamma=1.0, eps=.1, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((H, W, 4)); returns = []
    for _ in range(episodes):
        s, a, total, traj = START, None, 0, []
        a = policy(Q, s, eps, rng)
        for _ in range(500):
            s2, r, done = env_step(s, a)
            a2 = policy(Q, s2, eps, rng)
            total += r
            if method == "SARSA":
                Q[s][a] += alpha * (r + gamma * Q[s2][a2] * (not done) - Q[s][a])
            elif method == "Q-learning":
                Q[s][a] += alpha * (r + gamma * Q[s2].max() * (not done) - Q[s][a])
            else:
                traj.append((s, a, r))
            s, a = s2, a2
            if done:
                break
        if method == "Monte Carlo":                   # every-visit MC update at the end of the episode
            G = 0
            for (st, at, rt) in reversed(traj):
                G = rt + gamma * G
                Q[st][at] += 0.05 * (G - Q[st][at])
        returns.append(total)
    return Q, np.array(returns)

def greedy_path(Q):
    grid = [["." for _ in range(W)] for _ in range(H)]
    for c in range(1, 11): grid[3][c] = "C"
    s = START
    for _ in range(40):
        a = int(np.argmax(Q[s])); grid[s[0]][s[1]] = ARROWS[a]
        s, _, done = env_step(s, a)
        if done or s == START: break
    grid[3][11] = "G"
    return "\n".join("".join(row) for row in grid)

plt.figure(figsize=(8, 4.2))
for method in ["SARSA", "Q-learning", "Monte Carlo"]:
    runs = [train(method, seed=k) for k in range(10)]
    curve = np.mean([r for _, r in runs], axis=0)
    smooth = np.convolve(curve, np.ones(20) / 20, mode="valid")
    print(f"{method:<12} average return (last 100 episodes, while exploring): {curve[-100:].mean():7.1f}")
    plt.plot(smooth, label=method)
    if method != "Monte Carlo":
        print(f"greedy path learned by {method}:\n{greedy_path(runs[0][0])}\n")
plt.ylim(-150, 0); plt.xlabel("episode"); plt.ylabel("return per episode (smoothed)")
plt.title("Cliff Walking with epsilon = 0.1"); plt.legend()
plt.tight_layout(); plt.savefig("session-10_cliff_learning.png", dpi=110); plt.show()
