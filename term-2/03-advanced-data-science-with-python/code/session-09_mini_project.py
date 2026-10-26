"""Mini project: turn association rules into a 'frequently bought together' recommender with a fallback."""
import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

rng = np.random.default_rng(11)
families = {"stationery": ["notebook", "pen", "pencil", "eraser", "ruler"], "tech": ["laptop", "mouse", "laptop bag", "usb drive"],
            "phone": ["phone", "phone case", "charger", "screen guard"], "printing": ["printer ink", "a4 paper"]}
baskets = []
for _ in range(5000):
    fam = rng.choice(list(families), p=[.45, .2, .25, .1])
    items = list(rng.choice(families[fam], rng.integers(1, len(families[fam]) + 1), replace=False))
    if rng.random() < .15:
        items.append(rng.choice(sum(families.values(), [])))
    baskets.append(sorted({str(x) for x in items}))
te = TransactionEncoder(); onehot = pd.DataFrame(te.fit(baskets).transform(baskets), columns=te.columns_)
rules = association_rules(fpgrowth(onehot, min_support=0.01, use_colnames=True), metric="lift", min_threshold=1.2)
popular = onehot.mean().sort_values(ascending=False)

def recommend(cart, k=3):
    """Score candidate items by the best lift x confidence of rules whose antecedent is inside the cart."""
    cart = set(cart)
    ok = rules[rules.antecedents.apply(lambda a: a <= cart)]
    scores = {}
    for _, r in ok.iterrows():
        for item in r.consequents - cart:
            scores[item] = max(scores.get(item, 0), r.lift * r.confidence)
    recs = [i for i, _ in sorted(scores.items(), key=lambda x: -x[1])][:k]
    for item in popular.index:                       # fallback: most popular items not in the cart
        if len(recs) >= k: break
        if item not in cart and item not in recs:
            recs.append(item)
    return recs

for cart in [["laptop"], ["phone", "charger"], ["notebook"], ["printer ink"], ["a4 paper", "pen"]]:
    print(f"cart {cart} -> recommend {recommend(cart)}")
print(f"\nrules available: {len(rules)}")
