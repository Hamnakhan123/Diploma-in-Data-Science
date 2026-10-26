"""Mini project: plan a student's evening as an MDP over energy and hours left; solve with value iteration."""
import numpy as np

HOURS, ENERGY = 5, 4                       # hours left tonight (0..5), energy level (0..3)
ACTIONS = ["study", "rest", "scroll phone"]
gamma = 0.97

def transitions(h, e, a):
    """Return list of (prob, h', e', reward)."""
    if h == 0:
        return [(1.0, 0, e, 0.0)]
    if a == 0:   # study: learning depends on energy, costs energy
        learn = [0.0, 1.0, 2.5, 3.5][e]
        return [(0.8, h - 1, max(e - 1, 0), learn), (0.2, h - 1, e, learn)]
    if a == 1:   # rest: regain energy, small reward
        return [(0.9, h - 1, min(e + 1, ENERGY - 1), 0.3), (0.1, h - 1, e, 0.3)]
    return [(0.7, h - 1, e, 0.8), (0.3, h - 1, max(e - 1, 0), 0.8)]     # scrolling: fun now, sometimes tiring

V = np.zeros((HOURS + 1, ENERGY))
for _ in range(200):
    V_new = np.zeros_like(V)
    for h in range(HOURS + 1):
        for e in range(ENERGY):
            V_new[h, e] = max(sum(p * (r + gamma * V[h2, e2]) for p, h2, e2, r in transitions(h, e, a)) for a in range(3))
    if np.abs(V_new - V).max() < 1e-8: break
    V = V_new

print("optimal action by (hours left, energy):")
print("hours\\energy   " + "".join(f"{e:>14}" for e in range(ENERGY)))
for h in range(HOURS, 0, -1):
    best = [ACTIONS[int(np.argmax([sum(p * (r + gamma * V[h2, e2]) for p, h2, e2, r in transitions(h, e, a)) for a in range(3)]))] for e in range(ENERGY)]
    print(f"{h:>5}          " + "".join(f"{b:>14}" for b in best))
print(f"\nexpected value of an evening starting with 5 hours and energy 1: {V[5, 1]:.2f}")
