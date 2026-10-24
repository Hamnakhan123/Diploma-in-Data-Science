import random
random.seed(5)
nodes = ["tt1", "tt2", "tt3", "tt4"]
blocks = {f"split{i}": random.sample(nodes, 3) for i in range(1, 11)}
slots = {n: 2 for n in nodes}          # 2 map slots per TaskTracker
pending = list(blocks)
assign = []
while pending and any(slots.values()):
    for tt in nodes:
        if not slots[tt] or not pending:
            continue
        local = [s for s in pending if tt in blocks[s]]
        task = local[0] if local else pending[0]
        assign.append((tt, task, "node-local" if local else "remote"))
        pending.remove(task); slots[tt] -= 1

for tt, task, kind in assign:
    print(f"{tt} runs map({task})  [{kind}]")
print(f"\nlocal tasks: {sum(k == 'node-local' for *_, k in assign)}/{len(assign)}, waiting: {pending}")
