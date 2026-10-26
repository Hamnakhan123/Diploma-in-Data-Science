"""Mini project: apartment rent estimator with model comparison, tuning and a saved pipeline."""
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

rng = np.random.default_rng(3)
n = 1500
area = rng.choice(["DHA", "Gulshan", "Clifton", "North Nazimabad", "Saddar"], n)
size = rng.normal(1200, 350, n).clip(400, 3000)
beds = np.clip((size / 450).round() + rng.integers(-1, 2, n), 1, 6)
floor = rng.integers(0, 15, n)
lift = (floor > 3) | (rng.random(n) < .3)
area_mult = pd.Series(area).map({"DHA": 1.6, "Clifton": 1.5, "Gulshan": 1.0, "North Nazimabad": .9, "Saddar": .8}).values
rent = (30 * size * area_mult + 6000 * beds + 3000 * lift - 800 * floor * (~lift) + rng.normal(0, 5000, n)) / 1000
df = pd.DataFrame({"area": area, "size_sqft": size.round(), "bedrooms": beds, "floor": floor,
                   "lift": lift.astype(int), "rent_k": rent.round(1)})

X, y = df.drop(columns="rent_k"), df["rent_k"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.2, random_state=0)
prep = ColumnTransformer([("num", StandardScaler(), ["size_sqft", "bedrooms", "floor", "lift"]),
                          ("cat", OneHotEncoder(handle_unknown="ignore"), ["area"])])
models = {"ridge": RidgeCV(), "forest": RandomForestRegressor(200, random_state=0, n_jobs=-1),
          "boosting": GradientBoostingRegressor(random_state=0)}
for name, reg in models.items():
    mae = -cross_val_score(Pipeline([("prep", prep), ("reg", reg)]), X_tr, y_tr, cv=5,
                           scoring="neg_mean_absolute_error").mean()
    print(f"{name:<9} CV MAE: Rs {mae:.1f}k")

search = GridSearchCV(Pipeline([("prep", prep), ("reg", GradientBoostingRegressor(random_state=0))]),
                      {"reg__n_estimators": [100, 300], "reg__max_depth": [2, 3, 4]},
                      cv=5, scoring="neg_mean_absolute_error").fit(X_tr, y_tr)
print("best params:", search.best_params_)
print(f"test MAE: Rs {mean_absolute_error(y_te, search.predict(X_te)):.1f}k per month")
joblib.dump(search.best_estimator_, "rent_model.joblib")
loaded = joblib.load("rent_model.joblib")
flat = pd.DataFrame([{"area": "Gulshan", "size_sqft": 1100, "bedrooms": 3, "floor": 2, "lift": 0}])
print(f"estimated rent for a 1100 sqft 3-bed in Gulshan: Rs {loaded.predict(flat)[0]:.0f}k per month")
