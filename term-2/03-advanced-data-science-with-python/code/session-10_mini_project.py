"""Mini project: Q-learning robot in a warehouse grid with shelves, a charging penalty and a delivery goal."""
import numpy as np

layout = ["S....#....",
          ".##..#.##.",
          ".#.......#",
          ".#.###.#..",
          "....#..#.D"]
H, W = len(layout), len(layout[0])
MOVES = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
names = list(MOVES)
start = next((r, c) for r in range(H) for c in range(W) if layout[r][c] == "S")
dock = next((r, c) for r in range(H) for c in range(W) if layout[r][c] == "D")

def step(s, a):
    dr, dc = MOVES[names[a]]
    r, c = s[0] + dr, s[1] + dc
    if not (0 <= r < H and 0 <= c < W) or layout[r][c] == "#":
        return s, -5, False                       # bumped into a shelf or wall
    if (r, c) == dock:
        return (r, c), 50, True
    return (r, c), -1, False

rng = np.random.default_rng(0)
Q = np.zeros((H, W, 4))
alpha, gamma = 0.3, 0.95
for ep in range(3000):
    eps = max(0.02, 1 - ep / 1500)
    s = start
    for _ in range(200):
        a = rng.integers(4) if rng.random() < eps else int(np.argmax(Q[s]))
        s2, r, done = step(s, a)
        Q[s][a] += alpha * (r + gamma * Q[s2].max() * (not done) - Q[s][a])
        s = s2
        if done:
            break
    if ep in (0, 500, 1500, 2999):
        print(f"after episode {ep:>4}: value of start state = {Q[start].max():6.2f}")

grid = [list(row) for row in layout]
s, steps = start, 0
while s != dock and steps < 50:
    a = int(np.argmax(Q[s]))
    if s != start:
        grid[s[0]][s[1]] = "^v<>"[a]
    s, _, _ = step(s, a); steps += 1
print(f"\ngreedy route found in {steps} moves:")
print("\n".join("".join(row) for row in grid))
