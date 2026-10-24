cluster_containers = 200
queues = {"etl": (50, 100), "analytics": (30, 100), "adhoc": (20, 40)}  # (capacity %, max %)
demand = {"etl": 40, "analytics": 150, "adhoc": 120}

alloc = {q: min(demand[q], cluster_containers * c // 100) for q, (c, _) in queues.items()}
free = cluster_containers - sum(alloc.values())
print("guaranteed:", alloc, "| idle:", free)
for q, (c, mx) in sorted(queues.items(), key=lambda x: -demand[x[0]]):   # lend idle capacity (elasticity)
    extra = min(free, demand[q] - alloc[q], cluster_containers * mx // 100 - alloc[q])
    if extra > 0:
        alloc[q] += extra; free -= extra
print("after elasticity:", alloc, "| idle:", free)
