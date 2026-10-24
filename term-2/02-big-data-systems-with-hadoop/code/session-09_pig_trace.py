from collections import defaultdict
sales = [(1, "Karachi", 2500.0), (2, "Lahore", 800.0), (3, "Karachi", 1200.0),
         (4, "Islamabad", 3100.0), (5, "Lahore", 1500.0), (6, "Karachi", 400.0), (7, "Lahore", 2600.0)]
print("sales   :", len(sales), "tuples")

big = [t for t in sales if t[2] > 1000]                      # FILTER
print("big     :", big)

by_city = defaultdict(list)                                   # GROUP big BY city
for t in big:
    by_city[t[1]].append(t)
print("by_city :")
for city, bag in by_city.items():
    print(f"   ({city}, {{{', '.join('(' + ','.join(map(str, t)) + ')' for t in bag)}}})")

totals = [(c, len(bag), sum(t[2] for t in bag)) for c, bag in by_city.items()]   # FOREACH GENERATE
ranked = sorted(totals, key=lambda r: -r[2])                                      # ORDER BY DESC
print("ranked  :", ranked)
