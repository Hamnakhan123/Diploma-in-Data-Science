"""ADF test before and after differencing, and ACF/PACF plots of an AR(2) process."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

rng = np.random.default_rng(1)
# A random walk (non-stationary): a share price
walk = pd.Series(100 + np.cumsum(rng.normal(0, 1, 500)))
def adf(s, name):
    stat, p = adfuller(s.dropna())[:2]
    print(f"{name:<28} ADF stat {stat:7.2f}  p-value {p:.4f}  -> {'stationary' if p < .05 else 'NOT stationary'}")
adf(walk, "random walk (price)")
adf(walk.diff(), "first difference (returns)")

# An AR(2) process: y_t = 0.6 y_{t-1} - 0.3 y_{t-2} + e_t
e = rng.normal(0, 1, 600); ar = np.zeros(600)
for i in range(2, 600):
    ar[i] = 0.6 * ar[i - 1] - 0.3 * ar[i - 2] + e[i]
ar = pd.Series(ar[100:])
adf(ar, "AR(2) process")
print("ACF  lags 1-5:", acf(ar, nlags=5)[1:].round(2))
print("PACF lags 1-5:", pacf(ar, nlags=5)[1:].round(2), " <- cuts off after lag 2")

fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))
ax[0].plot(walk, color="#A78BFA"); ax[0].set_title("Random walk: not stationary")
plot_acf(ar, lags=20, ax=ax[1], title="ACF of AR(2): tails off")
plot_pacf(ar, lags=20, ax=ax[2], title="PACF of AR(2): cuts off after lag 2", method="ywm")
fig.tight_layout(); fig.savefig("session-05_acf_pacf.png", dpi=100); plt.show()
