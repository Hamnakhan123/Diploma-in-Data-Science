from collections import defaultdict

def run_mapreduce(splits, mapper, reducer, combiner=None, n_reducers=2, partitioner=None):
    partitioner = partitioner or (lambda k, r: sum(map(ord, str(k))) % r)
    shuffled = [defaultdict(list) for _ in range(n_reducers)]
    sent = 0
    for split in splits:                                   # one mapper per split
        local = defaultdict(list)
        for record in split:
            for k, v in mapper(record):
                local[k].append(v)
        if combiner:
            local = {k: [combiner(k, vs)] for k, vs in local.items()}
        for k, vs in local.items():
            sent += len(vs)
            shuffled[partitioner(k, n_reducers)][k].extend(vs)
    outputs = [[reducer(k, vs) for k, vs in sorted(part.items())] for part in shuffled]
    return outputs, sent

splits = [["the cat sat on the mat", "the dog sat"], ["the cat ran", "a dog and the cat"]]
mapper = lambda line: ((w, 1) for w in line.split())
reducer = lambda k, vs: (k, sum(vs))

for comb in [None, reducer]:
    out, sent = run_mapreduce(splits, mapper, reducer, combiner=(lambda k, vs: sum(vs)) if comb else None)
    print(f"combiner={'yes' if comb else 'no ':<3}  pairs sent over network = {sent}")
for i, part in enumerate(out):
    print(f"part-r-0000{i}: {part}")
