import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import cross_val_score

rows = """sunny,hot,high,FALSE,no
sunny,hot,high,TRUE,no
overcast,hot,high,FALSE,yes
rainy,mild,high,FALSE,yes
rainy,cool,normal,FALSE,yes
rainy,cool,normal,TRUE,no
overcast,cool,normal,TRUE,yes
sunny,mild,high,FALSE,no
sunny,cool,normal,FALSE,yes
rainy,mild,normal,FALSE,yes
sunny,mild,normal,TRUE,yes
overcast,mild,high,TRUE,yes
overcast,hot,normal,FALSE,yes
rainy,mild,high,TRUE,no"""
df = pd.DataFrame([r.split(",") for r in rows.split("\n")],
                  columns=["outlook", "temperature", "humidity", "windy", "play"])
X = pd.get_dummies(df.drop(columns="play")).astype(int)
y = df["play"]

tree = DecisionTreeClassifier(criterion="entropy", random_state=0).fit(X, y)
print(export_text(tree, feature_names=list(X.columns)))
scores = cross_val_score(DecisionTreeClassifier(criterion="entropy", random_state=0), X, y, cv=7)
print("7-fold CV accuracy:", scores.round(2), "mean =", round(scores.mean(), 3))
