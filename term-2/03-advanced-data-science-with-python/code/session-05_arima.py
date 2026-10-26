"""Grid-search ARIMA orders by AIC, check residuals with Ljung-Box, forecast with intervals."""
import itertools
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
warnings.filterwarnings("ignore")

rng = np.random.default_rng(2)
# Weekly demand with a gentle trend and AR(1) behaviour
n = 156
e = rng.normal(0, 8, n); y = np.zeros(n)
for i in range(1, n):
    y[i] = 0.7 * y[i - 1] + e[i]
y = pd.Series(500 + 1.5 * np.arange(n) + y, index=pd.date_range("2023-01-01", periods=n, freq="W"))
train, test = y[:-12], y[-12:]

results = []
for p, d, q in itertools.product(range(3), range(2), range(3)):
    try:
        m = ARIMA(train, order=(p, d, q), trend="t" if d == 1 else "ct").fit()
        results.append(((p, d, q), m.aic))
    except Exception:
        pass
results.sort(key=lambda r: r[1])
print("best 5 orders by AIC:")
for order, aic in results[:5]:
    print(f"  ARIMA{order}  AIC = {aic:.1f}")

best_order = results[0][0]
model = ARIMA(train, order=best_order, trend="t" if best_order[1] == 1 else "ct").fit()
lb = acorr_ljungbox(model.resid, lags=[10])
print(f"\nLjung-Box p-value on residuals: {lb['lb_pvalue'].iloc[0]:.3f}  (> 0.05 = residuals look like white noise)")
fc = model.get_forecast(12)
out = pd.DataFrame({"actual": test.round(0), "forecast": fc.predicted_mean.round(0)})
out[["low95", "high95"]] = fc.conf_int().round(0).values
print(out.head(6).to_string())
mae = (out.actual - out.forecast).abs().mean(); naive = (test - train.iloc[-1]).abs().mean()
print(f"\nMAE ARIMA = {mae:.1f}   MAE naive (last value) = {naive:.1f}")
