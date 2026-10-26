"""Create a monthly sales series with trend + yearly seasonality + noise and decompose it."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

rng = np.random.default_rng(0)
idx = pd.date_range("2016-01-01", periods=120, freq="MS")           # 10 years, month starts
t = np.arange(120)
season = 1 + 0.25 * np.sin(2 * np.pi * (t - 2) / 12)                  # summer peak
sales = (200 + 3 * t) * season * np.exp(rng.normal(0, 0.04, 120))     # multiplicative
y = pd.Series(sales.round(1), index=idx, name="ac_units")
print(y.head(14).to_string())

res = seasonal_decompose(y, model="multiplicative", period=12)
print("\nseasonal index by month (1.00 = average):")
print(res.seasonal[:12].round(3).set_axis([d.strftime("%b") for d in idx[:12]]).to_string())

fig = res.plot(); fig.set_size_inches(10, 7)
fig.suptitle("Multiplicative decomposition of monthly AC sales", y=1.0)
fig.tight_layout(); fig.savefig("session-05_decomposition.png", dpi=100); plt.show()
