import random
random.seed(11)
nodes = [f"dn{i}" for i in range(1, 7)]
block_map = {f"blk_{i}": random.sample(nodes, 3) for i in range(1, 11)}

def report(title):
    under = {b: r for b, r in block_map.items() if len(r) < 3}
    print(f"{title}: {len(block_map)} blocks, under-replicated = {len(under)}")
    return under

report("healthy cluster")
dead = "dn2"
print(f"\n*** no heartbeat from {dead} for 10 minutes -> marked dead ***")
for reps in block_map.values():
    if dead in reps:
        reps.remove(dead)
under = report("after failure")
for b, reps in sorted(under.items()):
    target = random.choice([n for n in nodes if n != dead and n not in reps])
    print(f"   copy {b} from {reps[0]} to {target}")
    reps.append(target)
report("\nafter re-replication")
