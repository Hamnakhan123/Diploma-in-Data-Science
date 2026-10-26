"""Compare linear and non-linear regressors with 5-fold cross-validation on a curved, interacting target."""
import numpy as np
from sklearn.model_selection import cross_validate, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

rng = np.random.default_rng(1)
n = 800
X = rng.uniform(0, 10, size=(n, 4))
# delivery time (min): grows with distance, worse in rush hour, interaction rain x distance
y = 10 + 3 * X[:, 0] + 8 * np.sin(X[:, 1]) + 0.8 * X[:, 0] * (X[:, 2] > 7) + rng.normal(0, 2, n)

models = {
    "Baseline (mean)": DummyRegressor(),
    "Linear regression": LinearRegression(),
    "Ridge": make_pipeline(StandardScaler(), Ridge(1.0)),
    "KNN (k=10)": make_pipeline(StandardScaler(), KNeighborsRegressor(10)),
    "Decision tree (depth 6)": DecisionTreeRegressor(max_depth=6, random_state=0),
    "Random forest": RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1),
    "Gradient boosting": GradientBoostingRegressor(random_state=0),
    "SVR (RBF)": make_pipeline(StandardScaler(), SVR(C=30, epsilon=0.5)),
}
cv = KFold(5, shuffle=True, random_state=0)
print(f"{'model':<25}{'MAE':>7}{'RMSE':>7}{'R2':>7}")
for name, m in models.items():
    r = cross_validate(m, X, y, cv=cv, scoring=("neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"))
    print(f"{name:<25}{-r['test_neg_mean_absolute_error'].mean():>7.2f}"
          f"{-r['test_neg_root_mean_squared_error'].mean():>7.2f}{r['test_r2'].mean():>7.3f}")
