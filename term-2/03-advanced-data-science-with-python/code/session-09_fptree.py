"""Build an FP-Tree by hand to see its structure, then mine with mlxtend's fpgrowth and compare with apriori."""
import time
from collections import Counter
import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth

baskets = [["tea", "sugar", "rusk"], ["tea", "sugar"], ["tea", "milk", "rusk"], ["bread", "eggs", "milk"],
           ["bread", "eggs"], ["tea", "sugar", "milk"], ["bread", "milk", "eggs", "tea"], ["tea", "rusk"],
           ["sugar", "tea", "rusk"], ["eggs", "bread", "butter"]]
MIN_COUNT = 3
freq = Counter(i for b in baskets for i in b)
order = {i: c for i, c in freq.items() if c >= MIN_COUNT}
print("item order by frequency:", sorted(order.items(), key=lambda x: -x[1]))

class Node:
    def __init__(self, item, parent):
        self.item, self.parent, self.count, self.children = item, parent, 0, {}

root = Node(None, None)
for b in baskets:
    path = sorted([i for i in b if i in order], key=lambda i: (-order[i], i))   # sort by frequency
    node = root
    for item in path:
        node = node.children.setdefault(item, Node(item, node))
        node.count += 1

def show(node, depth=0):
    for child in sorted(node.children.values(), key=lambda n: -n.count):
        print("  " * depth + f"{child.item}:{child.count}")
        show(child, depth + 1)
print("\nFP-Tree:"); show(root)

# mlxtend on a bigger synthetic data set: apriori vs fpgrowth speed
rng = np.random.default_rng(0)
items = [f"item{i:02d}" for i in range(40)]
pop = rng.dirichlet(np.ones(40) * .4)
big = [list(rng.choice(items, size=rng.integers(2, 12), replace=False, p=pop)) for _ in range(5000)]
te = TransactionEncoder(); onehot = pd.DataFrame(te.fit(big).transform(big), columns=te.columns_)
for name, fn in [("apriori", apriori), ("fpgrowth", fpgrowth)]:
    t0 = time.perf_counter(); res = fn(onehot, min_support=0.02, use_colnames=True); dt = time.perf_counter() - t0
    print(f"{name:<9} {len(res):>5} frequent itemsets in {dt:.2f} s")
