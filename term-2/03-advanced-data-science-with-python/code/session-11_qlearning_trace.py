"""Q-learning on the same slippery grid without knowing the model; trace one update and compare with Q*."""
import numpy as np

N, GOAL, TRAP, gamma, slip = 4, 15, 5, 0.95, 0.2
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]
def env_step(s, a, rng):
    r = rng.random()
    a = a if r < 1 - slip else (a + 1) % 4 if r < 1 - slip / 2 else (a + 3) % 4
    row, col = divmod(s, N); dr, dc = MOVES[a]
    s2 = min(max(row + dr, 0), N - 1) * N + min(max(col + dc, 0), N - 1)
    if s2 == GOAL: return s2, 10.0, True
    if s2 == TRAP: return s2, -10.0, True
    return s2, -0.5, False

rng = np.random.default_rng(0)
Q = np.zeros((N * N, 4)); alpha = 0.1
for ep in range(20000):
    s, eps = 0, max(0.05, 1 - ep / 8000)
    for t in range(100):
        a = rng.integers(4) if rng.random() < eps else int(np.argmax(Q[s]))
        s2, r, done = env_step(s, a, rng)
        target = r + gamma * Q[s2].max() * (not done)
        td_error = target - Q[s, a]
        if ep == 15000 and t < 3:
            print(f"ep {ep} step {t}: s={s} a={'^>v<'[a]} r={r:+.1f} s'={s2}  "
                  f"Q={Q[s, a]:.3f} target={target:.3f} TD error={td_error:+.3f}")
        Q[s, a] += alpha * td_error
        s = s2
        if done: break

print("\nlearned greedy policy:")
for row in range(N):
    print(" ".join("G" if row * N + c == GOAL else "X" if row * N + c == TRAP else "^>v<"[int(np.argmax(Q[row * N + c]))] for c in range(N)))
print("learned V(start) = max Q(0, a) =", Q[0].max().round(2))
