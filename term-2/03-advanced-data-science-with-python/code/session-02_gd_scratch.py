"""Linear regression with batch gradient descent, written from scratch with NumPy."""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
units = rng.uniform(100, 600, 80)                     # kWh used in a month
bill = 1500 + 42 * units + rng.normal(0, 1200, 80)    # true rule: Rs 1500 fixed + Rs 42 per unit

# Scale x so gradient descent behaves (mean 0, sd 1)
mu, sd = units.mean(), units.std()
x = (units - mu) / sd

w, b, lr = 0.0, 0.0, 0.1
history = []
for epoch in range(200):
    y_hat = w * x + b
    err = bill - y_hat
    loss = np.mean(err ** 2)
    dw = -2 * np.mean(x * err)
    db = -2 * np.mean(err)
    w, b = w - lr * dw, b - lr * db
    history.append(loss)
    if epoch in (0, 1, 5, 20, 50, 199):
        print(f"epoch {epoch:>3}: MSE = {loss:>14,.0f}  w = {w:8.1f}  b = {b:8.1f}")

# Convert back to the original units: bill = b0 + slope * units
slope = w / sd
b0 = b - slope * mu
print(f"\nlearned rule: bill = {b0:.0f} + {slope:.2f} x units   (true: 1500 + 42 x units)")
print(f"predicted bill for 350 units: Rs {b0 + slope * 350:,.0f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(history, color="#F43F5E"); ax[0].set_yscale("log")
ax[0].set_title("Loss falls as gradient descent learns"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("MSE (log scale)")
ax[1].scatter(units, bill, s=15, color="#A78BFA", label="houses")
grid = np.linspace(100, 600, 50)
ax[1].plot(grid, b0 + slope * grid, color="#FF6B1A", lw=2, label="fitted line")
ax[1].set_title("Fitted regression line"); ax[1].set_xlabel("units (kWh)"); ax[1].set_ylabel("bill (Rs)"); ax[1].legend()
fig.tight_layout(); fig.savefig("session-02_gradient_descent.png", dpi=110); plt.show()
