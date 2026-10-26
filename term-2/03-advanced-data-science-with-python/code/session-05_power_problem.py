"""Real-life problem: 12-month electricity demand forecast with SARIMA order search, validation and intervals."""
import itertools
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
warnings.filterwarnings("ignore")

rng = np.random.default_rng(42)
idx = pd.date_range("2015-01-01", periods=132, freq="MS"); t = np.arange(132)
summer = np.array([0.80, 0.82, 0.92, 1.05, 1.22, 1.30, 1.25, 1.20, 1.12, 0.98, 0.85, 0.79])
gwh = pd.Series((1500 * 1.045 ** (t / 12)) * summer[idx.month - 1] * np.exp(rng.normal(0, .03, 132)), index=idx)
gwh.iloc[63] *= 1.12                                              # one heat-wave month
train, test = gwh[:-12], gwh[-12:]
print(f"data: {gwh.index[0]:%b %Y} to {gwh.index[-1]:%b %Y}, {len(gwh)} months; average {gwh.mean():.0f} GWh")

# Search small SARIMA orders on log data by AIC
cands = []
for p, q, P, Q in itertools.product([0, 1, 2], [0, 1], [0, 1], [0, 1]):
    try:
        m = SARIMAX(np.log(train), order=(p, 1, q), seasonal_order=(P, 1, Q, 12)).fit(disp=False)
        cands.append((m.aic, (p, 1, q), (P, 1, Q, 12)))
    except Exception:
        pass
cands.sort()
aic, order, sorder = cands[0]
print(f"best SARIMA{order}{sorder} with AIC {aic:.1f}  (searched {len(cands)} models)")

sar = SARIMAX(np.log(train), order=order, seasonal_order=sorder).fit(disp=False)
f_sar = np.exp(sar.forecast(12))
f_hw = ExponentialSmoothing(initialization_method="estimated", endog=train, trend="add", seasonal="mul", seasonal_periods=12).fit().forecast(12)
f_sn = pd.Series(train[-12:].values * (train[-12:].sum() / train[-24:-12].sum()), index=test.index)  # naive + growth
for name, f in [("SARIMA", f_sar), ("Holt-Winters", f_hw), ("Seasonal naive + growth", f_sn)]:
    print(f"{name:<24} validation MAPE = {(np.abs(test - f) / test).mean() * 100:.2f}%")

final = SARIMAX(np.log(gwh), order=order, seasonal_order=sorder).fit(disp=False)
fc = final.get_forecast(12)
plan = pd.DataFrame({"forecast_GWh": np.exp(fc.predicted_mean), "upper95_GWh": np.exp(fc.conf_int().iloc[:, 1])}).round(0)
plan.index = plan.index.strftime("%b %Y")
print("\nPurchase plan for the next 12 months:\n", plan.to_string())
print(f"\nannual total: {plan.forecast_GWh.sum():,.0f} GWh (plan for up to {plan.upper95_GWh.sum():,.0f} GWh)")

ci = np.exp(fc.conf_int())
plt.figure(figsize=(10, 4))
plt.plot(gwh[-48:], color="black", label="history")
plt.plot(np.exp(fc.predicted_mean), color="#FF6B1A", lw=2, label="SARIMA forecast")
plt.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], color="#FCD34D", alpha=.4, label="95% interval")
plt.ylabel("GWh"); plt.title("Monthly electricity demand: next 12 months"); plt.legend()
plt.tight_layout(); plt.savefig("session-05_power_forecast.png", dpi=110); plt.show()
