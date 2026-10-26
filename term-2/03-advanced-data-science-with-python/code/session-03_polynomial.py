"""Fit polynomials of increasing degree and watch underfitting turn into overfitting."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(0)
# Air-conditioner sales vs temperature: rises then flattens (a curve, not a line)
temp = np.sort(rng.uniform(15, 45, 40))
sales = 50 + 30 * np.sin((temp - 15) / 18) + rng.normal(0, 4, 40)
temp_test = rng.uniform(15, 45, 200)
sales_test = 50 + 30 * np.sin((temp_test - 15) / 18) + rng.normal(0, 4, 200)

grid = np.linspace(15, 45, 300).reshape(-1, 1)
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
print("degree  train RMSE  test RMSE")
for i, d in enumerate([1, 3, 15]):
    model = make_pipeline(PolynomialFeatures(d), StandardScaler(), LinearRegression())
    model.fit(temp.reshape(-1, 1), sales)
    tr = mean_squared_error(sales, model.predict(temp.reshape(-1, 1))) ** .5
    te = mean_squared_error(sales_test, model.predict(temp_test.reshape(-1, 1))) ** .5
    print(f"{d:>5}   {tr:>9.2f}  {te:>9.2f}")
    ax[i].scatter(temp, sales, s=14, color="#A78BFA")
    ax[i].plot(grid, np.clip(model.predict(grid), 0, 120), color="#FF6B1A", lw=2)
    ax[i].set_title(f"degree {d}: " + ["underfit", "good fit", "overfit"][i]); ax[i].set_xlabel("temperature °C")
ax[0].set_ylabel("AC units sold")
fig.tight_layout(); fig.savefig("session-03_polynomial_fits.png", dpi=110); plt.show()
