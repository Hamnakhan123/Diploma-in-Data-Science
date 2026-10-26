"""PCA on the 64-pixel digits data, then Apriori association rules with mlxtend."""
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

X, y = load_digits(return_X_y=True)                    # 1797 images, 8x8 = 64 features
pca = PCA().fit(X)
cum = pca.explained_variance_ratio_.cumsum()
for n in [2, 10, 20, 30]:
    print(f"{n:>2} components keep {cum[n - 1]:.1%} of the variance")
print("components needed for 90%:", int((cum < .90).sum()) + 1, "of 64")

baskets = [["tea", "sugar", "rusk"], ["tea", "sugar"], ["tea", "milk", "rusk"], ["bread", "eggs", "milk"],
           ["bread", "eggs"], ["tea", "sugar", "milk"], ["bread", "milk", "eggs", "tea"], ["tea", "rusk"],
           ["sugar", "tea", "rusk"], ["eggs", "bread", "butter"]]
te = TransactionEncoder()
onehot = pd.DataFrame(te.fit(baskets).transform(baskets), columns=te.columns_)
freq = apriori(onehot, min_support=0.3, use_colnames=True)
rules = association_rules(freq, metric="lift", min_threshold=1.2)
rules["rule"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s))) + " -> " + rules["consequents"].apply(lambda s: ", ".join(sorted(s)))
print("\n", rules[["rule", "support", "confidence", "lift"]].round(2).sort_values("lift", ascending=False).to_string(index=False))
