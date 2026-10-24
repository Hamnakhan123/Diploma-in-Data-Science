from collections import defaultdict
customers = [(1, "Ayesha", "Karachi"), (2, "Bilal", "Lahore"), (3, "Sana", "Multan")]
orders = [(101, 1, 2500), (102, 2, 900), (103, 1, 4100), (104, 4, 700)]

def map_customers(r):  yield r[0], ("C", r[1], r[2])
def map_orders(r):     yield r[1], ("O", r[0], r[2])

shuffle = defaultdict(list)
for r in customers:
    for k, v in map_customers(r): shuffle[k].append(v)
for r in orders:
    for k, v in map_orders(r): shuffle[k].append(v)

print("inner join (cust_id, name, city, order_id, amount):")
for key in sorted(shuffle):
    cs = [v for v in shuffle[key] if v[0] == "C"]
    os_ = [v for v in shuffle[key] if v[0] == "O"]
    for c in cs:
        for o in os_:
            print(" ", (key, c[1], c[2], o[1], o[2]))
print("records shuffled over the network:", sum(len(v) for v in shuffle.values()))
