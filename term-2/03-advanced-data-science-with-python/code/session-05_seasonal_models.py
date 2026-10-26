"""Compare seasonal naive, Holt, Holt-Winters and SARIMA on a 12-month hold-out; plot the forecasts."""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
warnings.filterwarnings("ignore")

rng = np.random.default_rng(0)
idx = pd.date_range("2016-01-01", periods=120, freq="MS"); t = np.arange(120)
y = pd.Series((200 + 3 * t) * (1 + 0.25 * np.sin(2 * np.pi * (t - 2) / 12)) * np.exp(rng.normal(0, .04, 120)), index=idx)
train, test = y[:-12], y[-12:]

fc = {
    "Seasonal naive": pd.Series(train[-12:].values, index=test.index),
    "Simple exp. smoothing": SimpleExpSmoothing(train, initialization_method="estimated").fit().forecast(12),
    "Holt (trend)": ExponentialSmoothing(initialization_method="estimated", endog=train, trend="add").fit().forecast(12),
    "Holt-Winters (mult.)": ExponentialSmoothing(initialization_method="estimated", endog=train, trend="add", seasonal="mul", seasonal_periods=12).fit().forecast(12),
    "SARIMA(1,1,1)(0,1,1,12)": SARIMAX(np.log(train), order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
                               .forecast(12).pipe(np.exp),
}
naive_mae = (test - fc["Seasonal naive"]).abs().mean()
print(f"{'model':<26}{'MAE':>7}{'RMSE':>7}{'MAPE %':>8}{'MASE':>6}")
for name, f in fc.items():
    err = test - f
    print(f"{name:<26}{err.abs().mean():>7.1f}{np.sqrt((err ** 2).mean()):>7.1f}"
          f"{(err.abs() / test).mean() * 100:>8.1f}{err.abs().mean() / naive_mae:>6.2f}")

hw = ExponentialSmoothing(initialization_method="estimated", endog=train, trend="add", seasonal="mul", seasonal_periods=12).fit()
# Parameters near 0 mean the pattern is so regular that the model barely updates its initial level, trend and season
print(f"\nHolt-Winters smoothing parameters: alpha={hw.params['smoothing_level']:.2f}, "
      f"beta={hw.params['smoothing_trend']:.2f}, gamma={hw.params['smoothing_seasonal']:.2f}")

plt.figure(figsize=(10, 4))
plt.plot(y[-48:], color="black", label="actual")
for name, c in [("Seasonal naive", "#999999"), ("Holt (trend)", "#A78BFA"), ("Holt-Winters (mult.)", "#FF6B1A"), ("SARIMA(1,1,1)(0,1,1,12)", "#F43F5E")]:
    plt.plot(fc[name], ls="--", lw=2, color=c, label=name)
plt.axvline(test.index[0], color="grey", lw=.8); plt.legend(fontsize=8); plt.ylabel("units")
plt.title("12-month hold-out: seasonal models beat trend-only Holt")
plt.tight_layout(); plt.savefig("session-05_forecast_comparison.png", dpi=110); plt.show()
