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

print("min_support  itemsets  rules(conf>=0.6)")
for ms in [0.1, 0.2, 0.3, 0.4, 0.5]:
    fi = apriori(onehot, min_support=ms, use_colnames=True)
    r = association_rules(fi, metric="confidence", min_threshold=0.6) if len(fi) else []
    print(f"{ms:>10}  {len(fi):>8}  {len(r):>6}")
