import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

transactions = [
    ["bread", "butter", "milk"], ["bread", "butter", "jam"], ["bread", "milk", "tea"],
    ["milk", "eggs", "tea"], ["bread", "butter", "milk", "jam"], ["bread", "butter", "jam", "tea"],
    ["milk", "eggs"], ["bread", "butter", "milk", "eggs"], ["tea"], ["bread", "butter", "jam"],
]
te = TransactionEncoder()
onehot = pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)

frequent = apriori(onehot, min_support=0.3, use_colnames=True)
print("Frequent itemsets (support >= 0.3):")
print(frequent.sort_values("support", ascending=False).to_string(index=False))

rules = association_rules(frequent, metric="confidence", min_threshold=0.8)
cols = ["antecedents", "consequents", "support", "confidence", "lift"]
rules = rules[cols].sort_values("lift", ascending=False)
rules["antecedents"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
rules["consequents"] = rules["consequents"].apply(lambda s: ", ".join(sorted(s)))
print("\nRules (confidence >= 0.8):")
print(rules.round(2).to_string(index=False))
