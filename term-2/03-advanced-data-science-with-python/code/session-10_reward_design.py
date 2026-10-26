"""Same Q-learning agent, three reward functions on a 6x6 grid: sparse, step penalty, and a hackable bonus."""
import numpy as np

SIZE, GOAL, COIN = 6, (5, 5), (0, 2)
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def step(s, a):
    r, c = s[0] + MOVES[a][0], s[1] + MOVES[a][1]
    return (min(max(r, 0), SIZE - 1), min(max(c, 0), SIZE - 1))

def reward(design, s, s2):
    if design == "sparse":
        return 1.0 if s2 == GOAL else 0.0
    if design == "step penalty":
        return 10.0 if s2 == GOAL else -1.0
    # 'hackable': +2 every time the agent stands on a coin square that never runs out
    return 10.0 if s2 == GOAL else (2.0 if s2 == COIN else -1.0)

def train(design, episodes=1500, alpha=.5, gamma=.95, eps=.2, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((SIZE, SIZE, 4))
    for _ in range(episodes):
        s = (0, 0)
        for _ in range(100):
            a = rng.integers(4) if rng.random() < eps else int(np.argmax(Q[s]))
            s2 = step(s, a); r = reward(design, s, s2)
            target = r if s2 == GOAL else r + gamma * Q[s2].max()
            Q[s][a] += alpha * (target - Q[s][a]); s = s2
            if s == GOAL:
                break
    # evaluate greedy policy
    s, path = (0, 0), [(0, 0)]
    for _ in range(40):
        s = step(s, int(np.argmax(Q[s]))); path.append(s)
        if s == GOAL:
            break
    return path

for design in ["sparse", "step penalty", "hackable"]:
    path = train(design)
    reached = path[-1] == GOAL
    coin_visits = sum(p == COIN for p in path)
    print(f"{design:<13} reached goal: {reached!s:<5}  steps: {len(path) - 1:>2}  coin visits: {coin_visits}")
