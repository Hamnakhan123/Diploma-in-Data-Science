"""Effect of max_depth on a decision tree, cost-complexity pruning, and printing the rules."""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

data = load_breast_cancer()
X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target, test_size=.3, random_state=0, stratify=data.target)

print("depth  leaves  train acc  test acc")
for d in [1, 2, 3, 5, 8, None]:
    t = DecisionTreeClassifier(max_depth=d, random_state=0).fit(X_tr, y_tr)
    print(f"{str(d):>5}  {t.get_n_leaves():>6}  {t.score(X_tr, y_tr):>9.3f}  {t.score(X_te, y_te):>8.3f}")

path = DecisionTreeClassifier(random_state=0).cost_complexity_pruning_path(X_tr, y_tr)
print("\nccp_alpha  leaves  test acc")
for a in path.ccp_alphas[::4]:
    t = DecisionTreeClassifier(ccp_alpha=a, random_state=0).fit(X_tr, y_tr)
    print(f"{a:9.4f}  {t.get_n_leaves():>6}  {t.score(X_te, y_te):.3f}")

small = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X_tr, y_tr)
print("\n" + export_text(small, feature_names=list(data.feature_names)))
