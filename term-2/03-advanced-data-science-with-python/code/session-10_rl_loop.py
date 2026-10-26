"""A tiny corridor environment, a random policy and a smart policy, and the discounted return."""
import numpy as np

class Corridor:
    """States 0..5. Start at 0. Reaching 5 gives +10 and ends the episode. Each step costs -1."""
    def __init__(self, length=6):
        self.length = length
    def reset(self):
        self.s = 0
        return self.s
    def step(self, action):                       # action: 0 = left, 1 = right
        self.s = max(0, min(self.length - 1, self.s + (1 if action == 1 else -1)))
        done = self.s == self.length - 1
        return self.s, (10 if done else -1), done

def run(policy, env, gamma=0.9, seed=0, max_steps=200):
    rng = np.random.default_rng(seed)
    s, rewards = env.reset(), []
    for _ in range(max_steps):
        s, r, done = env.step(policy(s, rng))
        rewards.append(r)
        if done:
            break
    G = sum(gamma ** t * r for t, r in enumerate(rewards))
    return len(rewards), sum(rewards), G

env = Corridor()
random_policy = lambda s, rng: rng.integers(2)
right_policy = lambda s, rng: 1
for name, pol in [("random", random_policy), ("always right", right_policy)]:
    stats = np.array([run(pol, env, seed=k) for k in range(200)])
    print(f"{name:<13} avg steps {stats[:, 0].mean():6.1f}  avg total reward {stats[:, 1].mean():7.1f}  avg discounted return {stats[:, 2].mean():6.2f}")

rewards = [-1, -1, -1, -1, 10]
for g in [0.0, 0.5, 0.9, 1.0]:
    print(f"gamma = {g:<4} -> return of the optimal episode = {sum(g ** t * r for t, r in enumerate(rewards)):6.2f}")
