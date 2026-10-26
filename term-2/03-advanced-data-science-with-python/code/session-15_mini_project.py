"""Mini project: a reusable explain() function that prints a plain-language SHAP explanation for any house."""
import warnings
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")

rng = np.random.default_rng(3)
n = 2500
houses = pd.DataFrame({"area_sqft": rng.normal(1400, 450, n).clip(400, 4000).round(), "bedrooms": rng.integers(1, 6, n),
                       "age_years": rng.integers(0, 45, n), "distance_to_centre_km": rng.exponential(7, n).clip(.5, 35).round(1),
                       "has_parking": rng.integers(0, 2, n), "corner_plot": (rng.random(n) < .15).astype(int)})
price = (0.085 * houses.area_sqft + 6 * houses.bedrooms - 0.9 * houses.age_years - 22 * np.log(houses.distance_to_centre_km)
         + 12 * houses.has_parking + 9 * houses.corner_plot + 80 + rng.normal(0, 10, n))
X_tr, X_te, y_tr, y_te = train_test_split(houses, price, test_size=.2, random_state=0)
model = RandomForestRegressor(300, min_samples_leaf=3, random_state=0, n_jobs=-1).fit(X_tr, y_tr)
explainer = shap.TreeExplainer(model)
print(f"model R2 on test houses: {model.score(X_te, y_te):.3f}")

def explain(house: pd.DataFrame, top=4):
    sv = explainer(house)
    base = float(np.ravel(explainer.expected_value)[0]); pred = model.predict(house)[0]
    contrib = pd.Series(sv.values[0], index=house.columns).sort_values(key=np.abs, ascending=False)
    print(f"\nEstimated price: Rs {pred:.1f} lakh (average house: Rs {base:.1f} lakh)")
    for feat in contrib.index[:top]:
        v = contrib[feat]
        print(f"  {'+' if v > 0 else '-'} Rs {abs(v):5.1f} lakh because {feat.replace('_', ' ')} = {house.iloc[0][feat]}")
    print(f"  (all contributions sum to {contrib.sum():+.1f}; base + sum = {base + contrib.sum():.1f})")

explain(pd.DataFrame([{"area_sqft": 2400, "bedrooms": 4, "age_years": 3, "distance_to_centre_km": 2.5, "has_parking": 1, "corner_plot": 1}]))
explain(pd.DataFrame([{"area_sqft": 800, "bedrooms": 2, "age_years": 35, "distance_to_centre_km": 25.0, "has_parking": 0, "corner_plot": 0}]))
