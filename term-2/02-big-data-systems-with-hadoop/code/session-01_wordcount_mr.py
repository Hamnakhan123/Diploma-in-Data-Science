from collections import defaultdict

splits = [                                   # imagine each line is a block on a different node
    "big data needs big clusters",
    "hadoop stores big data in hdfs",
    "yarn shares the cluster",
]

def mapper(line):
    for word in line.split():
        yield word, 1

def reducer(word, counts):
    return word, sum(counts)

# map phase (runs in parallel on the nodes that hold each split)
mapped = [pair for line in splits for pair in mapper(line)]
print("map output (first 6):", mapped[:6])

# shuffle and sort: group all values by key
groups = defaultdict(list)
for k, v in mapped:
    groups[k].append(v)

# reduce phase
result = [reducer(k, v) for k, v in sorted(groups.items())]
print("reduce output:")
for word, n in sorted(result, key=lambda x: (-x[1], x[0])):
    print(f"  {word:<9}{n}")
