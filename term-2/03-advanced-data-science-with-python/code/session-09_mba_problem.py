"""Real-life problem: end-to-end market basket analysis from raw invoice lines to recommended actions."""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import fpgrowth, association_rules

rng = np.random.default_rng(7)
catalogue = ["milk", "bread", "eggs", "butter", "tea", "sugar", "rusk", "rice", "lentils", "cooking oil",
             "onions", "tomatoes", "yogurt", "biscuits", "soft drink", "chips", "nappies", "baby wipes", "shampoo", "soap"]
patterns = [(["milk", "bread", "eggs"], .20), (["tea", "sugar"], .18), (["tea", "rusk"], .12), (["rice", "lentils", "cooking oil"], .10),
            (["onions", "tomatoes"], .22), (["soft drink", "chips"], .12), (["nappies", "baby wipes"], .05), (["shampoo", "soap"], .06)]
lines = []
for inv in range(8000):
    items = set()
    for combo, p in patterns:
        if rng.random() < p:
            items.update(c for c in combo if rng.random() < .85)       # not always the full combo
    items.update(rng.choice(catalogue, rng.integers(1, 4), replace=False))
    for it in items:
        lines.append((f"INV{inv:05d}", str(it), int(rng.integers(1, 4))))
raw = pd.DataFrame(lines, columns=["invoice", "item", "qty"])
cancel = raw.sample(frac=.01, random_state=1).index
raw.loc[cancel, "qty"] = -raw.loc[cancel, "qty"]                       # returns / cancellations
print("raw invoice lines:", len(raw), "| cancelled lines:", (raw.qty < 0).sum())

# 1. Clean: drop cancelled lines, build a one-hot basket matrix
clean = raw[raw.qty > 0]
basket = clean.assign(v=True).pivot_table(index="invoice", columns="item", values="v", aggfunc="any", fill_value=False)
print("basket matrix:", basket.shape)

# 2. Mine and 3. filter rules
freq = fpgrowth(basket, min_support=0.02, use_colnames=True)
rules = association_rules(freq, metric="lift", min_threshold=1.3)
rules = rules[(rules.confidence >= .3) & (rules.consequents.apply(len) == 1) & (rules.antecedents.apply(len) <= 2)].copy()
# (single-item consequents, at most two antecedents: longer rules mostly glue two unrelated patterns together)
rules["rule"] = rules.antecedents.apply(lambda s: " + ".join(sorted(s))) + "  ->  " + rules.consequents.apply(lambda s: " + ".join(sorted(s)))
# 4. Remove redundancy: keep the single best rule per unordered item set
rules["itemset"] = rules.apply(lambda r: frozenset(r.antecedents | r.consequents), axis=1)
best = rules.sort_values("lift", ascending=False).drop_duplicates("itemset")
print(f"\nfrequent itemsets: {len(freq)} | rules after filters: {len(rules)} | after removing duplicates: {len(best)}")
print(best[["rule", "support", "confidence", "lift"]].head(10).round(3).to_string(index=False))

weak = association_rules(freq, metric="lift", min_threshold=0).query("confidence > 0.3 and lift < 1.1")
weak = weak.assign(rule=weak.antecedents.apply(lambda s: "+".join(sorted(s))) + " -> " + weak.consequents.apply(lambda s: "+".join(sorted(s))))
print("\nhigh-confidence but NOT useful (lift ~ 1, consequent is just popular):")
print(weak.sort_values("confidence", ascending=False)[["rule", "confidence", "lift"]].head(4).round(2).to_string(index=False))

top = best.head(12)
plt.figure(figsize=(9, 5))
sns.barplot(data=top, y="rule", x="lift", hue="confidence", palette="flare", dodge=False)
plt.axvline(1, color="grey", ls="--"); plt.title("Top association rules (lift; colour = confidence)")
plt.tight_layout(); plt.savefig("session-09_mba_rules.png", dpi=110); plt.show()
