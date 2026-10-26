"""Mini project: 14-day restaurant demand forecast and staffing plan (Holt-Winters vs SARIMAX vs naive)."""
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
warnings.filterwarnings("ignore")

rng = np.random.default_rng(9)
idx = pd.date_range("2025-09-01", periods=420, freq="D")
dow_effect = np.array([0.8, 0.75, 0.8, 0.9, 1.25, 1.45, 1.3])        # Mon..Sun
holiday = pd.Series(0, index=idx); holiday[idx[::45]] = 1               # a public holiday roughly every 45 days
covers = (120 + 0.05 * np.arange(420)) * dow_effect[idx.dayofweek] * (1 + 0.35 * holiday.values) \
         * np.exp(rng.normal(0, .07, 420))
y = pd.Series(covers.round(), index=idx)
train, test = y[:-14], y[-14:]
hol_tr, hol_te = holiday[:-14].to_frame("hol"), holiday[-14:].to_frame("hol")

f_naive = pd.Series(train[-7:].tolist() * 2, index=test.index)
f_hw = ExponentialSmoothing(initialization_method="estimated", endog=train, trend="add", seasonal="mul", seasonal_periods=7).fit().forecast(14)
sx = SARIMAX(np.log(train), exog=hol_tr, order=(1, 0, 1), seasonal_order=(1, 1, 1, 7), trend="c").fit(disp=False)
f_sx = np.exp(sx.forecast(14, exog=hol_te))
for name, f in [("Seasonal naive", f_naive), ("Holt-Winters", f_hw), ("SARIMAX + holiday", f_sx)]:
    print(f"{name:<18} MAE = {np.abs(test - f).mean():5.1f} covers/day")

# Refit SARIMAX on all data (near-tied with Holt-Winters, and it can use the holiday flag) and plan 14 days (no holidays ahead)
future_idx = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=14, freq="D")
final = SARIMAX(np.log(y), exog=holiday.to_frame("hol"), order=(1, 0, 1), seasonal_order=(1, 1, 1, 7), trend="c").fit(disp=False)
fc = np.exp(final.forecast(14, exog=pd.DataFrame({"hol": 0}, index=future_idx)))
plan = pd.DataFrame({"day": future_idx.strftime("%a %d %b"), "covers": fc.round().astype(int).values})
plan["waiters_needed"] = np.ceil(plan["covers"] / 25).astype(int)     # one waiter per 25 covers
print("\n" + plan.to_string(index=False))
