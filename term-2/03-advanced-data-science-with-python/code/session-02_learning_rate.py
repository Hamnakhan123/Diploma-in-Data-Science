"""Effect of the learning rate, and the same model in scikit-learn (closed form and SGD)."""
import numpy as np
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(0)
units = rng.uniform(100, 600, 80)
bill = 1500 + 42 * units + rng.normal(0, 1200, 80)
x = (units - units.mean()) / units.std()

def train(lr, epochs=50):
    w = b = 0.0
    for _ in range(epochs):
        err = bill - (w * x + b)
        w += lr * 2 * np.mean(x * err); b += lr * 2 * np.mean(err)
    return np.mean((bill - (w * x + b)) ** 2)

for lr in [0.001, 0.01, 0.1, 0.5, 1.05]:
    print(f"learning rate {lr:<6} -> MSE after 50 epochs: {train(lr):,.0f}")

X = units.reshape(-1, 1)                         # scikit-learn wants a 2-D X
ols = LinearRegression().fit(X, bill)
sgd = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, random_state=0)).fit(X, bill)
print(f"\nLinearRegression (exact least squares): bill = {ols.intercept_:.0f} + {ols.coef_[0]:.2f} x units")
print(f"SGDRegressor prediction for 350 units: Rs {sgd.predict([[350]])[0]:,.0f}")
print(f"LinearRegression prediction for 350  : Rs {ols.predict([[350]])[0]:,.0f}")
