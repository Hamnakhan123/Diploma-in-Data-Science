import math, random
random.seed(7)

BLOCK_MB, REPLICATION = 128, 3
datanodes = ["dn1", "dn2", "dn3", "dn4", "dn5"]

def split_file(name, size_mb):
    n = math.ceil(size_mb / BLOCK_MB)
    blocks = []
    for i in range(n):
        size = min(BLOCK_MB, size_mb - i * BLOCK_MB)
        replicas = random.sample(datanodes, REPLICATION)
        blocks.append((f"{name}#blk_{i+1}", size, replicas))
    return blocks

for fname, size in [("calls_jan.csv", 300), ("logo.png", 2), ("cctv_day1.mp4", 1000)]:
    blocks = split_file(fname, size)
    print(f"{fname}: {size} MB -> {len(blocks)} block(s), raw storage used = {size*REPLICATION} MB")
    for b, s, reps in blocks:
        print(f"   {b:<22} {s:>4} MB  on {', '.join(reps)}")
