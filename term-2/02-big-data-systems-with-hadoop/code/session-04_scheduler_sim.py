def simulate(policy, capacity=10):
    jobs = {"big_batch": {"arrive": 0, "left": 100, "done": None},
            "small_query": {"arrive": 1, "left": 5, "done": None}}
    t = 0
    while any(j["left"] for j in jobs.values()):
        active = [n for n, j in jobs.items() if j["arrive"] <= t and j["left"]]
        share = {n: 0 for n in active}
        free = capacity
        if policy == "FIFO":
            for n in sorted(active, key=lambda n: jobs[n]["arrive"]):
                share[n] = min(free, jobs[n]["left"]); free -= share[n]
        else:  # Fair: split equally, give leftovers to whoever still needs them
            while free and any(share[n] < jobs[n]["left"] for n in active):
                for n in active:
                    if free and share[n] < jobs[n]["left"]:
                        share[n] += 1; free -= 1
        t += 1
        for n, s in share.items():
            jobs[n]["left"] -= s
            if jobs[n]["left"] == 0 and jobs[n]["done"] is None:
                jobs[n]["done"] = t
    return {n: j["done"] - j["arrive"] for n, j in jobs.items()}

for policy in ["FIFO", "Fair"]:
    r = simulate(policy)
    print(f"{policy:<5} turnaround  big_batch={r['big_batch']:>3}  small_query={r['small_query']:>3}")
