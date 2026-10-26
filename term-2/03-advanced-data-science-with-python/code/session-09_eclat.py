"""Eclat: vertical tid-lists and depth-first intersection to find frequent itemsets."""
baskets = [{"tea", "sugar", "rusk"}, {"tea", "sugar"}, {"tea", "milk", "rusk"}, {"bread", "eggs", "milk"},
           {"bread", "eggs"}, {"tea", "sugar", "milk"}, {"bread", "milk", "eggs", "tea"}, {"tea", "rusk"},
           {"sugar", "tea", "rusk"}, {"eggs", "bread", "butter"}]
MIN_COUNT = 3                                      # = support 0.3 with 10 transactions

tid = {}
for t, b in enumerate(baskets, 1):
    for item in b:
        tid.setdefault(item, set()).add(t)
print("vertical format (tid-lists):")
for item in sorted(tid):
    print(f"  {item:<7} {sorted(tid[item])}")

results = []
def eclat(prefix, candidates):
    """candidates: list of (item, tidset) that can extend prefix, sorted."""
    for i, (item, tids) in enumerate(candidates):
        if len(tids) < MIN_COUNT:
            continue
        itemset = prefix + [item]
        results.append((itemset, len(tids)))
        # extend with later items by intersecting tid-lists
        ext = [(other, tids & otids) for other, otids in candidates[i + 1:]]
        eclat(itemset, ext)

eclat([], sorted(tid.items()))
print("\nfrequent itemsets (count >= 3):")
for itemset, c in sorted(results, key=lambda r: (len(r[0]), -r[1])):
    print(f"  {', '.join(itemset):<22} count={c}  support={c / len(baskets):.1f}")
