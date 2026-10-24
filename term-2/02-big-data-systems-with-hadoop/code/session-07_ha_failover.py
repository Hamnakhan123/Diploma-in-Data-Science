journal = {"jn1": [], "jn2": [], "jn3": []}
state = {"nn1": "active", "nn2": "standby"}
standby_ns = []

def write_edit(op):
    acks = 0
    for jn, log in journal.items():
        if jn != "jn3" or len(log) < 2:        # jn3 goes offline after 2 edits
            log.append(op); acks += 1
    ok = acks >= 2                              # majority of 3
    print(f"edit {op:<22} acks={acks}/3 -> {'committed' if ok else 'FAILED'}")

def standby_tail():
    longest = max(journal.values(), key=len)
    standby_ns.extend(longest[len(standby_ns):])

for op in ["mkdir /data", "create /data/a.csv", "create /data/b.csv", "delete /data/a.csv"]:
    write_edit(op)
    standby_tail()

print("\n*** ZKFC on nn1: health check failed, ZooKeeper lock released ***")
state["nn1"] = "fenced"
state["nn2"] = "active"
print("NameNode states:", state)
print("nn2 namespace edits:", standby_ns)
