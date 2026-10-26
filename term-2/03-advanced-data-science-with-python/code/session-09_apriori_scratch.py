"""A small, readable Apriori implementation: join, prune, count, and rule generation."""
from itertools import combinations

baskets = [{"tea", "sugar", "rusk"}, {"tea", "sugar"}, {"tea", "milk", "rusk"}, {"bread", "eggs", "milk"},
           {"bread", "eggs"}, {"tea", "sugar", "milk"}, {"bread", "milk", "eggs", "tea"}, {"tea", "rusk"},
           {"sugar", "tea", "rusk"}, {"eggs", "bread", "butter"}]
N, MIN_SUP, MIN_CONF = len(baskets), 0.3, 0.6

def support(itemset):
    return sum(itemset <= b for b in baskets) / N          # <= means 'is a subset of'

items = sorted({i for b in baskets for i in b})
L = [{frozenset([i]) for i in items if support({i}) >= MIN_SUP}]
print("L1:", sorted("".join(sorted(s)) for s in L[0]))
k = 2
while L[-1]:
    prev = L[-1]
    candidates = {a | b for a in prev for b in prev if len(a | b) == k}                         # join
    candidates = {c for c in candidates if all(frozenset(s) in prev for s in combinations(c, k - 1))}  # prune
    Lk = {c for c in candidates if support(c) >= MIN_SUP}                                       # count
    print(f"C{k}: {len(candidates)} candidates -> L{k}: {[sorted(c) for c in Lk]}")
    L.append(Lk); k += 1

print("\nRules (confidence >= 0.6):")
for level in L[1:]:
    for itemset in level:
        for r in range(1, len(itemset)):
            for A in map(frozenset, combinations(itemset, r)):
                B = itemset - A
                conf = support(itemset) / support(A)
                if conf >= MIN_CONF:
                    lift = conf / support(B)
                    print(f"  {sorted(A)} -> {sorted(B)}  support={support(itemset):.2f}  confidence={conf:.2f}  lift={lift:.2f}")
