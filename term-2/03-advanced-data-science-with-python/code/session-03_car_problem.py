"""Real-life problem: used-car price prediction with a full preprocessing pipeline and model comparison."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

rng = np.random.default_rng(7)
n = 2500
make = rng.choice(["Suzuki", "Toyota", "Honda", "Kia", "Hyundai"], n, p=[.35, .25, .2, .1, .1])
base = pd.Series(make).map({"Suzuki": 22, "Toyota": 48, "Honda": 45, "Kia": 52, "Hyundai": 50}).values
year = rng.integers(2008, 2026, n)
engine = np.where(make == "Suzuki", rng.choice([660, 1000, 1300], n), rng.choice([1300, 1500, 1800, 2000], n))
mileage = ((2026 - year) * rng.normal(14000, 4000, n)).clip(1000)
auto = rng.random(n) < 0.55
city = rng.choice(["Karachi", "Lahore", "Islamabad", "Peshawar"], n)
price = base * (0.92 ** (2026 - year)) * (engine / 1300) ** .6 * (1 + .08 * auto) * np.exp(-mileage / 900000) \
        * np.where(city == "Islamabad", 1.04, 1.0) * np.exp(rng.normal(0, .08, n))      # lakh Rs
cars = pd.DataFrame({"make": make, "year": year, "engine_cc": engine, "mileage_km": mileage.round(-2),
                     "transmission": np.where(auto, "Automatic", "Manual"), "city": city, "price_lakh": price.round(2)})
cars.loc[rng.choice(n, 80, replace=False), "mileage_km"] = np.nan       # missing values
cars.loc[rng.choice(n, 5, replace=False), "price_lakh"] = 0.01          # data-entry errors

cars = cars[cars["price_lakh"] > 1]                                      # 1. remove impossible prices
print("rows after cleaning:", len(cars), "| missing mileage:", cars["mileage_km"].isna().sum())
X, y = cars.drop(columns="price_lakh"), cars["price_lakh"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

num, cat = ["year", "engine_cc", "mileage_km"], ["make", "transmission", "city"]
prep = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])

def model(reg):            # predict log(price), report in lakh: handles skew and multiplicative effects
    return TransformedTargetRegressor(Pipeline([("prep", prep), ("reg", reg)]), func=np.log, inverse_func=np.exp)

candidates = {"Baseline (mean)": DummyRegressor(),
              "Ridge": model(RidgeCV(alphas=np.logspace(-3, 3, 13))),
              "Lasso": model(LassoCV(cv=5, random_state=0)),
              "Gradient boosting": model(HistGradientBoostingRegressor(random_state=0))}
cv = KFold(5, shuffle=True, random_state=1)
for name, m in candidates.items():
    s = -cross_val_score(m, X_tr, y_tr, cv=cv, scoring="neg_mean_absolute_error").mean()
    print(f"{name:<18} CV MAE = {s:6.2f} lakh")

best = candidates["Gradient boosting"].fit(X_tr, y_tr)
pred = best.predict(X_te)
print(f"\nTEST: MAE = {mean_absolute_error(y_te, pred):.2f} lakh, R2 = {r2_score(y_te, pred):.3f}")
err = pd.DataFrame({"make": X_te["make"], "abs_err": np.abs(y_te - pred)})
print("MAE by make (lakh):\n", err.groupby("make")["abs_err"].mean().round(2).to_string())
sample = pd.DataFrame({"make": ["Toyota"], "year": [2019], "engine_cc": [1300], "mileage_km": [85000],
                       "transmission": ["Automatic"], "city": ["Karachi"]})
print(f"\nSuggested price for a 2019 Toyota 1300cc automatic, 85,000 km: Rs {best.predict(sample)[0]:.1f} lakh")

plt.figure(figsize=(5.5, 5))
plt.scatter(y_te, pred, s=10, alpha=.5, color="#A78BFA"); m_ = y_te.max()
plt.plot([0, m_], [0, m_], color="#FF6B1A"); plt.xlabel("actual price (lakh)"); plt.ylabel("predicted price (lakh)")
plt.title("Used cars: predicted vs actual (test set)"); plt.tight_layout()
plt.savefig("session-03_car_predictions.png", dpi=110); plt.show()
