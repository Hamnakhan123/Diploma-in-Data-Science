"""Value iteration and policy iteration on a 4x4 slippery grid world with a trap; compare iterations and time."""
import time
import numpy as np

N, GOAL, TRAP = 4, 15, 5
ACTIONS = ["^", ">", "v", "<"]
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]
gamma, slip = 0.95, 0.2                          # 20% of the time the agent slips sideways

def move(s, a):
    r, c = divmod(s, N); dr, dc = MOVES[a]
    r2, c2 = min(max(r + dr, 0), N - 1), min(max(c + dc, 0), N - 1)
    return r2 * N + c2

# Build P[s, a] as a list of (prob, next_state, reward, done)
P = {}
for s in range(N * N):
    for a in range(4):
        outcomes = []
        for prob, aa in [(1 - slip, a), (slip / 2, (a + 1) % 4), (slip / 2, (a + 3) % 4)]:
            s2 = s if s in (GOAL, TRAP) else move(s, aa)
            rew = 0 if s in (GOAL, TRAP) else (10 if s2 == GOAL else -10 if s2 == TRAP else -0.5)
            outcomes.append((prob, s2, rew, s2 in (GOAL, TRAP)))
        P[s, a] = outcomes

def q_from_v(V, s):
    return np.array([sum(p * (r + gamma * V[s2] * (not d)) for p, s2, r, d in P[s, a]) for a in range(4)])

def value_iteration(theta=1e-6):
    V, sweeps = np.zeros(N * N), 0
    while True:
        sweeps += 1
        V_new = np.array([q_from_v(V, s).max() for s in range(N * N)])
        if np.max(np.abs(V_new - V)) < theta:
            return V_new, np.array([q_from_v(V_new, s).argmax() for s in range(N * N)]), sweeps
        V = V_new

def policy_iteration():
    pi, rounds, evals = np.zeros(N * N, int), 0, 0
    while True:
        rounds += 1
        V = np.zeros(N * N)
        while True:                                   # policy evaluation
            evals += 1
            V_new = np.array([q_from_v(V, s)[pi[s]] for s in range(N * N)])
            if np.max(np.abs(V_new - V)) < 1e-6: break
            V = V_new
        new_pi = np.array([q_from_v(V, s).argmax() for s in range(N * N)])   # improvement
        if np.array_equal(new_pi, pi):
            return V, pi, rounds, evals
        pi = new_pi

def show(pi, V):
    for r in range(N):
        row = ""
        for c in range(N):
            s = r * N + c
            row += " G " if s == GOAL else " X " if s == TRAP else f" {ACTIONS[pi[s]]} "
        print(row + "   " + " ".join(f"{V[r * N + c]:6.2f}" for c in range(N)))

t0 = time.perf_counter(); V1, pi1, sweeps = value_iteration(); t1 = time.perf_counter() - t0
t0 = time.perf_counter(); V2, pi2, rounds, evals = policy_iteration(); t2 = time.perf_counter() - t0
print(f"value iteration : {sweeps} sweeps, {t1 * 1000:.0f} ms")
print(f"policy iteration: {rounds} improvement rounds ({evals} evaluation sweeps in total), {t2 * 1000:.0f} ms")
both_optimal = all(np.isclose(q_from_v(V1, s)[pi2[s]], q_from_v(V1, s).max()) for s in range(N * N))
print("max value difference:", float(np.abs(V1 - V2).max().round(5)),
      "| identical arrows:", bool(np.array_equal(pi1, pi2)), "| both policies optimal:", both_optimal)
# (arrows can differ only where two actions are exactly tied, e.g. the goal and trap cells)
print("\noptimal policy (X = trap, G = goal) and state values:")
show(pi1, V1)
