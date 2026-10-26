"""Real-life problem: push-notification optimisation with Thompson sampling vs a fixed A/B test (plus context)."""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(2026)
messages = ["Free delivery today", "20% off biryani", "Your favourites are waiting", "Lunch in 25 min", "Try something new"]
rate = {"student": np.array([.050, .072, .030, .061, .040]),        # true conversion rates (unknown)
        "family":  np.array([.048, .055, .066, .040, .035])}
USERS = 20000
segments = rng.choice(["student", "family"], USERS, p=[.55, .45])

def simulate(method):
    a_s = {g: np.ones(5) for g in rate}; b_s = {g: np.ones(5) for g in rate}
    shown = np.zeros(5, int); orders = 0; regret = 0
    for i, g in enumerate(segments):
        ctx = g if method == "Thompson (contextual)" else "student"        # non-contextual: one shared model
        if method == "A/B test (uniform)":
            a = i % 5
        else:
            a = int(np.argmax(rng.beta(a_s[ctx], b_s[ctx])))
        r = rng.random() < rate[g][a]
        a_s[ctx][a] += r; b_s[ctx][a] += 1 - r
        shown[a] += 1; orders += r; regret += rate[g].max() - rate[g][a]
    return orders, regret, shown

print(f"{'method':<24}{'orders':>8}{'regret':>9}   share of traffic per message")
results = {}
for m in ["A/B test (uniform)", "Thompson (one model)", "Thompson (contextual)"]:
    o, reg, shown = simulate(m)
    results[m] = shown / USERS
    print(f"{m:<24}{o:>8}{reg:>9.1f}   {np.round(shown / USERS, 2)}")

best_all = (rate["student"] * .55 + rate["family"] * .45)
print("\nbest single message overall:", messages[int(best_all.argmax())])
print("best for students:", messages[int(rate['student'].argmax())], "| best for families:", messages[int(rate['family'].argmax())])

x = np.arange(5)
plt.figure(figsize=(9, 4))
for k, (m, share) in enumerate(results.items()):
    plt.bar(x + (k - 1) * .27, share, width=.27, label=m)
plt.xticks(x, [m.replace(" ", "\n", 1) for m in messages], fontsize=8); plt.ylabel("share of notifications")
plt.title("Where each method sent its 20,000 notifications"); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig("session-10_push_traffic.png", dpi=110); plt.show()
