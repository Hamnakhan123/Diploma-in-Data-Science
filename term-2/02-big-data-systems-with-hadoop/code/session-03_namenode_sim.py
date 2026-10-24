import math, random
random.seed(3)

racks = {"rack1": ["dn1", "dn2", "dn3"], "rack2": ["dn4", "dn5", "dn6"]}
rack_of = {dn: r for r, dns in racks.items() for dn in dns}
BLOCK = 128

def place(writer):
    first = writer
    other_rack = [r for r in racks if r != rack_of[first]][0]
    second = random.choice(racks[other_rack])
    third = random.choice([d for d in racks[other_rack] if d != second])
    return [first, second, third]

namespace = {}          # file -> list of block ids
block_map = {}          # block id -> list of DataNodes
next_id = 1000
for fname, size, writer in [("/data/calls.csv", 400, "dn1"), ("/data/tx.parquet", 200, "dn5")]:
    namespace[fname] = []
    for _ in range(math.ceil(size / BLOCK)):
        block_map[next_id] = place(writer)
        namespace[fname].append(next_id)
        next_id += 1

for f, blocks in namespace.items():
    print(f)
    for b in blocks:
        reps = block_map[b]
        print(f"   blk_{b}: {reps}  racks={sorted({rack_of[d] for d in reps})}")
