from collections import defaultdict
docs = {
    "d1": "hadoop stores big data",
    "d2": "spark processes big data in memory",
    "d3": "hive runs sql on hadoop",
}
# map
pairs = [(w, d) for d, text in docs.items() for w in set(text.split())]
# shuffle + reduce
index = defaultdict(set)
for w, d in pairs:
    index[w].add(d)
for w in sorted(index):
    print(f"{w:<10} -> {sorted(index[w])}")

query = ["big", "hadoop"]
hits = set.intersection(*(index[w] for w in query))
print(f"\nquery {query} -> {sorted(hits)}")
