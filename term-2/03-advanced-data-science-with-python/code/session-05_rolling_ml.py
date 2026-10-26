"""Rolling-origin CV with TimeSeriesSplit: Holt-Winters vs gradient boosting on lag features."""
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
warnings.filterwarnings("ignore")

rng = np.random.default_rng(5)
idx = pd.date_range("2024-01-01", periods=730, freq="D"); t = np.arange(730)
temp = 28 + 7 * np.sin(2 * np.pi * (t - 100) / 365) + rng.normal(0, 1.5, 730)        # exogenous driver
load = 900 + 0.3 * t + 25 * (temp - 28) + np.where(idx.dayofweek == 6, -120, 0) + rng.normal(0, 25, 730)
df = pd.DataFrame({"load": load, "temp": temp}, index=idx)
for L in [1, 7, 14]:
    df[f"lag_{L}"] = df["load"].shift(L)
df["dow"] = df.index.dayofweek
df["roll7"] = df["load"].shift(1).rolling(7).mean()
df = df.dropna()
features = ["temp", "lag_1", "lag_7", "lag_14", "dow", "roll7"]

tscv = TimeSeriesSplit(n_splits=4, test_size=30)
print("fold  train days  HW MAE  GBM MAE  seasonal-naive MAE")
for k, (tr, te) in enumerate(tscv.split(df), 1):
    train, test = df.iloc[tr], df.iloc[te]
    hw = ExponentialSmoothing(initialization_method="estimated", endog=train["load"], trend="add", seasonal="add", seasonal_periods=7).fit().forecast(len(test))
    gbm = HistGradientBoostingRegressor(random_state=0).fit(train[features], train["load"])
    p = gbm.predict(test[features])            # one-step-ahead: lags are known each morning
    sn = test["lag_7"]
    print(f"{k:>4}  {len(train):>10}  {np.mean(np.abs(test.load - hw.values)):>6.1f}  "
          f"{np.mean(np.abs(test.load - p)):>7.1f}  {np.mean(np.abs(test.load - sn)):>18.1f}")
