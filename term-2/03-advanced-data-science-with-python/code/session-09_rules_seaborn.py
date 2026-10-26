"""Full rule table (support, confidence, lift, leverage, conviction) and three Seaborn charts."""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

rng = np.random.default_rng(3)
# A bakery-cafe: some items are bought together on purpose
combos = [(["chai", "rusk"], .22), (["coffee", "croissant"], .15), (["bread", "butter", "eggs"], .12),
          (["chai", "samosa"], .14), (["juice", "sandwich"], .10)]
singles = ["chai", "coffee", "bread", "cake", "juice", "cookies", "samosa", "water"]
baskets = []
for _ in range(3000):
    b = set()
    for combo, p in combos:
        if rng.random() < p:
            b.update(combo)
    b.update(rng.choice(singles, rng.integers(0, 3), replace=False))
    if b:
        baskets.append(sorted(str(x) for x in b))          # plain str (not numpy str) for mlxtend
te = TransactionEncoder(); onehot = pd.DataFrame(te.fit(baskets).transform(baskets), columns=te.columns_)
print("transactions:", len(onehot), "| items:", onehot.shape[1])

freq = fpgrowth(onehot, min_support=0.03, use_colnames=True)
rules = association_rules(freq, metric="lift", min_threshold=1.0)
# keep simple one-item -> one-item rules: long rules that glue two unrelated patterns together get inflated lifts
rules = rules[(rules.antecedents.apply(len) == 1) & (rules.consequents.apply(len) == 1)].copy()
rules["rule"] = rules.antecedents.apply(lambda s: "+".join(sorted(s))) + " -> " + rules.consequents.apply(lambda s: "+".join(sorted(s)))
top = rules.sort_values("lift", ascending=False).head(8)
print(top[["rule", "support", "confidence", "lift", "leverage", "conviction"]].round(3).to_string(index=False))

sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
item_sup = onehot.mean().sort_values(ascending=False)
sns.barplot(x=item_sup.values, y=item_sup.index, ax=ax[0], color="#FF6B1A"); ax[0].set_title("Item support"); ax[0].set_xlabel("share of baskets")
sns.scatterplot(data=rules, x="support", y="confidence", hue="lift", size="lift", palette="flare", sizes=(20, 250), ax=ax[1])
ax[1].set_title("Rules: support vs confidence (colour and size = lift)")
co = onehot.T.astype(int) @ onehot.astype(int)
lift_m = (co / len(onehot)) / np.outer(item_sup.reindex(co.index), item_sup.reindex(co.columns))
lift_m = lift_m.mask(np.eye(len(lift_m), dtype=bool))       # hide the diagonal (item with itself)
sns.heatmap(lift_m, cmap="rocket_r", annot=True, fmt=".1f", annot_kws={"size": 7}, ax=ax[2], cbar_kws={"label": "lift"})
ax[2].set_title("Pairwise lift matrix")
fig.tight_layout(); fig.savefig("session-09_rules_seaborn.png", dpi=100); plt.show()
