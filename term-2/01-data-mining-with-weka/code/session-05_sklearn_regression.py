import pandas as pd
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = [
 (1200,3,10,"Gulshan",180),(900,2,15,"Korangi",95),(2000,4,5,"DHA",420),(1500,3,8,"Gulshan",230),
 (1800,4,12,"DHA",370),(800,2,20,"Korangi",80),(2500,5,3,"DHA",560),(1100,3,18,"Gulshan",160),
 (1300,3,6,"Korangi",135),(1600,3,9,"Gulshan",250),(2200,4,7,"DHA",470),(1000,2,14,"Korangi",105),
 (1400,3,11,"Gulshan",205),(3000,5,2,"DHA",690),(950,2,25,"Korangi",85),(1700,4,10,"Gulshan",265)]
df = pd.DataFrame(data, columns=["area_sqft", "bedrooms", "age_years", "location", "price_lakh"])
X = pd.get_dummies(df.drop(columns="price_lakh"), drop_first=True).astype(float)
y = df["price_lakh"]

models = {
    "Linear regression": LinearRegression(),
    "Regression tree": DecisionTreeRegressor(max_depth=3, random_state=0),
    "k-NN (k=3)": make_pipeline(StandardScaler(), KNeighborsRegressor(3)),
    "SVR (linear)": make_pipeline(StandardScaler(), SVR(kernel="linear", C=100)),
}
cv = KFold(4, shuffle=True, random_state=1)
print(f"{'model':<20}{'MAE':>8}{'RMSE':>8}{'R2':>7}")
for name, m in models.items():
    pred = cross_val_predict(m, X, y, cv=cv)
    rmse = mean_squared_error(y, pred) ** 0.5
    print(f"{name:<20}{mean_absolute_error(y, pred):>8.1f}{rmse:>8.1f}{r2_score(y, pred):>7.3f}")

lr = LinearRegression().fit(X, y)
print("\nCoefficients of the full linear model:")
for col, coef in zip(X.columns, lr.coef_):
    print(f"  {col:<18}{coef:>9.3f}")
print(f"  intercept         {lr.intercept_:>9.3f}")
print("\nPredicted price for 1500 sqft, 3 bed, 5 years, DHA:",
      round(lr.predict(pd.DataFrame([[1500, 3, 5, 1, 0]], columns=X.columns))[0], 1), "lakh")
