from collections import defaultdict
students = [("Ali", "Karachi", 82), ("Sara", "Lahore", 91), ("Omar", "Karachi", 77),
            ("Hina", "Lahore", None), ("Zain", "Karachi", 95)]
fees = [("Karachi", 45000), ("Lahore", 42000), ("Quetta", 30000)]

def fmt_bag(bag):
    return "{" + ", ".join("(" + ",".join("" if v is None else str(v) for v in t) + ")" for t in bag) + "}"

group = defaultdict(list)                                         # GROUP students BY city
for t in students:
    group[t[1]].append(t)
print("GROUP students BY city:")
for k, bag in group.items():
    print(f"  ({k}, {fmt_bag(bag)})")

print("\nFOREACH g GENERATE group, COUNT(students.score), AVG(students.score):")
for k, bag in group.items():
    scores = [t[2] for t in bag if t[2] is not None]             # nulls are skipped
    print(f"  ({k}, {len(scores)}, {sum(scores)/len(scores):.1f})")

print("\nCOGROUP students BY city, fees BY city:")
keys = sorted({t[1] for t in students} | {f[0] for f in fees})
for k in keys:
    print(f"  ({k}, {fmt_bag([t for t in students if t[1] == k])}, {fmt_bag([f for f in fees if f[0] == k])})")

print("\nFLATTEN back to flat tuples:")
for k, bag in group.items():
    for t in bag:
        print(f"  ({k},{t[0]})")
