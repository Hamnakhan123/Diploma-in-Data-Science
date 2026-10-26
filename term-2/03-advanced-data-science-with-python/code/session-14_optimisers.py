"""Compare gradient descent, momentum and Adam on an elongated valley (Rosenbrock-like) loss surface."""
import numpy as np
import matplotlib.pyplot as plt

def loss(w):            # a long, narrow, curved valley; minimum at (1, 1)
    x, y = w
    return (1 - x) ** 2 + 20 * (y - x ** 2) ** 2
def grad(w):
    x, y = w
    return np.array([-2 * (1 - x) - 80 * x * (y - x ** 2), 40 * (y - x ** 2)])

def run(method, lr, steps=600):
    w = np.array([-1.5, 2.0]); v = np.zeros(2); m = np.zeros(2); s = np.zeros(2); path = [w.copy()]
    for t in range(1, steps + 1):
        g = grad(w)
        if method == "gradient descent":
            w = w - lr * g
        elif method == "momentum":
            v = 0.9 * v + g; w = w - lr * v
        elif method == "Adam":
            m = 0.9 * m + 0.1 * g; s = 0.999 * s + 0.001 * g ** 2
            mh, sh = m / (1 - 0.9 ** t), s / (1 - 0.999 ** t)
            w = w - lr * mh / (np.sqrt(sh) + 1e-8)
        path.append(w.copy())
    return np.array(path)

configs = [("gradient descent", 0.005), ("momentum", 0.002), ("Adam", 0.05)]
xx, yy = np.meshgrid(np.linspace(-2, 2, 200), np.linspace(-1, 3, 200))
plt.figure(figsize=(7, 5))
plt.contour(xx, yy, (1 - xx) ** 2 + 20 * (yy - xx ** 2) ** 2, levels=np.logspace(-1, 3, 15), cmap="Greys", linewidths=.6)
for (name, lr), c in zip(configs, ["#A78BFA", "#F43F5E", "#FF6B1A"]):
    p = run(name, lr)
    print(f"{name:<17} lr={lr:<6} final point ({p[-1, 0]:.3f}, {p[-1, 1]:.3f})  loss {loss(p[-1]):.5f}  "
          f"steps to loss<0.01: {next((i for i, w in enumerate(p) if loss(w) < .01), 'not reached')}")
    plt.plot(p[:, 0], p[:, 1], color=c, lw=1.6, label=name)
plt.scatter([1], [1], marker="*", s=200, color="gold", zorder=5, label="minimum")
plt.legend(); plt.title("Optimiser paths in a narrow curved valley"); plt.xlabel("w1"); plt.ylabel("w2")
plt.tight_layout(); plt.savefig("session-14_optimisers.png", dpi=110); plt.show()
