"""Real-life problem: DQN controller for office air-conditioning vs a rule-based thermostat."""
import random
from collections import deque
import numpy as np
import torch
from torch import nn
torch.manual_seed(0); random.seed(0); torch.set_num_threads(2)
rng = np.random.default_rng(0)

ACTIONS = ["off", "eco", "boost"]
COOL = [0.0, 1.0, 2.2]           # degrees removed per 15 min
POWER = [0.0, 1.0, 2.6]          # kWh per 15 min
STEPS = 40                       # 8 am to 6 pm

def outdoor(t):                  # hot afternoon peak
    return 31 + 7 * np.sin(np.pi * t / STEPS)

class Office:
    def reset(self):
        self.t, self.temp = 0, 27.0 + rng.normal(0, .5)
        return self.state()
    def state(self):
        return np.array([(self.temp - 24) / 4, (outdoor(self.t) - 32) / 5, self.t / STEPS], dtype=np.float32)
    def step(self, a):
        drift = 0.12 * (outdoor(self.t) - self.temp) + 0.25              # heat from outside + people/computers
        self.temp += drift - COOL[a] + rng.normal(0, .1)
        self.t += 1
        discomfort = max(0, abs(self.temp - 24) - 1)                      # within 23-25 °C is fine
        reward = -(2.0 * discomfort + 0.4 * POWER[a])
        return self.state(), reward, self.t >= STEPS, discomfort, POWER[a]

def qnet():
    return nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3))

env, online, target = Office(), qnet(), qnet()
target.load_state_dict(online.state_dict())
opt = torch.optim.Adam(online.parameters(), lr=1e-3)
buffer, gamma, batch = deque(maxlen=20000), 0.95, 64
for episode in range(300):
    s, done, eps = env.reset(), False, max(0.05, 1 - episode / 150)
    while not done:
        if random.random() < eps:
            a = random.randrange(3)
        else:
            with torch.no_grad():
                a = int(online(torch.tensor(s)).argmax())
        s2, r, done, _, _ = env.step(a)
        buffer.append((s, a, r, s2, done)); s = s2
        if len(buffer) >= 500:
            S, A, R, S2, Dn = map(np.array, zip(*random.sample(buffer, batch)))
            S, S2 = torch.tensor(S), torch.tensor(S2)
            q = online(S).gather(1, torch.tensor(A).unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                y = torch.tensor(R, dtype=torch.float32) + gamma * target(S2).max(1).values * (1 - torch.tensor(Dn, dtype=torch.float32))
            loss = nn.functional.smooth_l1_loss(q, y)
            opt.zero_grad(); loss.backward(); opt.step()
    if episode % 10 == 0:
        target.load_state_dict(online.state_dict())          # refresh the target network

def evaluate(policy, days=30):
    tot_r = tot_d = tot_e = 0
    for _ in range(days):
        s, done = env.reset(), False
        while not done:
            s, r, done, disc, kwh = env.step(policy(s))
            tot_r += r; tot_d += disc; tot_e += kwh
    return tot_r / days, tot_d / days, tot_e / days

def rule(s):                                   # classic thermostat: boost above 25.5, eco above 24.5, else off
    temp = s[0] * 4 + 24
    return 2 if temp > 25.5 else 1 if temp > 24.5 else 0
def dqn(s):
    with torch.no_grad():
        return int(online(torch.tensor(s)).argmax())

print(f"{'controller':<22}{'reward/day':>11}{'discomfort/day':>16}{'energy kWh/day':>16}")
for name, pol in [("rule-based thermostat", rule), ("always eco", lambda s: 1), ("DQN agent", dqn)]:
    r, d, e = evaluate(pol)
    print(f"{name:<22}{r:>11.1f}{d:>16.2f}{e:>16.1f}")

env.reset(); env.temp = 27.0
plan = []
s = env.state()
for t in range(STEPS):
    a = dqn(s); plan.append(ACTIONS[a][0].upper())
    s, *_ = env.step(a)
print("\nDQN actions over the day (O=off, E=eco, B=boost):", "".join(plan))
