import random
random.seed(1)
records = [(random.randint(1, 999), f"student{i:02d}") for i in range(18)]   # (score, name)

sample = sorted(k for k, _ in random.sample(records, 6))     # sample keys to pick split points
cuts = [sample[2], sample[4]]                                # 3 reducers -> 2 split points
def range_partition(key):
    return 0 if key < cuts[0] else 1 if key < cuts[1] else 2

parts = {0: [], 1: [], 2: []}
for k, v in records:
    parts[range_partition(k)].append((k, v))
print("split points:", cuts)
for r in parts:
    parts[r].sort()                                          # the shuffle sorts each reducer's input
    print(f"part-r-0000{r}: {[k for k, _ in parts[r]]}")
merged = [k for r in parts for k, _ in parts[r]]
print("concatenated output is globally sorted:", merged == sorted(merged))
