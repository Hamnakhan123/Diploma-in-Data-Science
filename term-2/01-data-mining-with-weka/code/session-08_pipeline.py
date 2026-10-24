import io
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix

rows = """57,yearly,2800,1,fibre,no
55,yearly,800,2,fibre,no
60,monthly,2200,5,fibre,yes
7,monthly,900,2,dsl,yes
16,monthly,2300,0,fibre,yes
53,monthly,1400,0,dsl,no
33,yearly,1600,3,fibre,no
44,yearly,2400,1,fibre,no
36,monthly,1800,3,dsl,yes
33,monthly,2900,1,fibre,yes
25,monthly,3800,1,dsl,yes
32,yearly,1200,3,fibre,no
15,monthly,2500,3,fibre,yes
43,yearly,900,2,dsl,no
44,monthly,1900,0,dsl,no
44,monthly,2100,2,dsl,no
55,yearly,2400,1,dsl,no
35,yearly,3200,0,fibre,no
59,yearly,800,3,fibre,no
7,monthly,1300,2,fibre,yes
14,yearly,2200,3,dsl,no
57,monthly,3600,1,fibre,no
33,monthly,3800,5,fibre,yes
11,monthly,900,2,dsl,yes
11,yearly,2700,2,dsl,no
6,monthly,1300,3,fibre,yes
42,yearly,3300,2,fibre,no
16,monthly,1600,0,dsl,no
19,monthly,2600,1,fibre,yes
52,yearly,2500,2,dsl,no
15,yearly,3300,1,fibre,no
10,yearly,1100,2,dsl,no
50,yearly,3200,0,dsl,no
16,monthly,2700,4,dsl,yes
28,monthly,3600,0,dsl,yes
10,yearly,2600,1,fibre,no
36,monthly,1100,0,dsl,no
32,monthly,2000,1,fibre,no
52,monthly,2200,2,fibre,no
55,monthly,1200,0,fibre,no
44,monthly,1600,2,fibre,no
53,yearly,2000,0,dsl,no
33,yearly,3400,1,dsl,no
49,monthly,1400,3,fibre,no
28,yearly,2800,5,fibre,yes
36,yearly,3200,1,fibre,no
21,monthly,3500,1,dsl,yes
2,yearly,3400,1,dsl,no
8,yearly,1200,2,fibre,yes
7,monthly,1100,2,fibre,yes
46,yearly,1400,0,fibre,no
7,monthly,1800,2,dsl,yes
16,monthly,3700,1,fibre,yes
60,monthly,2400,0,dsl,no
57,yearly,1300,1,fibre,no
43,monthly,2800,3,fibre,no
33,yearly,3900,2,dsl,no
33,monthly,3700,1,fibre,yes
55,yearly,2200,5,dsl,no
20,yearly,1300,0,dsl,no"""
cols = ["tenure_months", "contract", "monthly_bill", "complaint_calls", "internet", "churn"]
df = pd.read_csv(io.StringIO(rows), header=None, names=cols)
X, y = df.drop(columns="churn"), (df["churn"] == "yes").astype(int)

prep = ColumnTransformer([
    ("num", StandardScaler(), ["tenure_months", "monthly_bill", "complaint_calls"]),
    ("cat", OneHotEncoder(drop="first"), ["contract", "internet"]),
])
pipe = Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=1000))])
cv = StratifiedKFold(5, shuffle=True, random_state=0)
grid = GridSearchCV(pipe, {"model__C": [0.1, 1, 10], "model__class_weight": [None, "balanced"]},
                    scoring="recall", cv=cv).fit(X, y)
print("Best settings:", grid.best_params_)

pred = cross_val_predict(grid.best_estimator_, X, y, cv=cv)
print(classification_report(y, pred, target_names=["stay", "churn"], digits=3))
tn, fp, fn, tp = confusion_matrix(y, pred).ravel()

# business value: each saved churner is worth Rs 30,000 of future revenue, each call costs Rs 500,
# and a call convinces about half of real churners to stay
value = tp * 0.5 * 30000 - (tp + fp) * 500
print(f"Calls made: {tp + fp}, churners reached: {tp}, missed: {fn}")
print(f"Estimated campaign value: Rs {value:,.0f}")

model = grid.best_estimator_.fit(X, y)
names = model.named_steps["prep"].get_feature_names_out()
coefs = pd.Series(model.named_steps["model"].coef_[0], index=names).sort_values()
print("\nCoefficients (positive = more likely to churn):")
print(coefs.round(2).to_string())

new = pd.DataFrame([[3, "monthly", 3500, 4, "fibre"], [48, "yearly", 1500, 0, "dsl"], [14, "monthly", 2200, 2, "dsl"]],
                   columns=cols[:-1])
print("\nChurn probability for new customers:", model.predict_proba(new)[:, 1].round(2).tolist())
