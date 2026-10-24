import sqlite3, random, os, tempfile
random.seed(2)
con = sqlite3.connect(":memory:")
con.execute("CREATE TABLE orders(id INTEGER PRIMARY KEY, customer_id INT, amount REAL, city TEXT)")
cities = ["Karachi", "Lahore", "Islamabad", "Peshawar", "Quetta"]
con.executemany("INSERT INTO orders VALUES (?,?,?,?)",
                [(i, random.randint(1, 50), round(random.uniform(500, 20000), 2), random.choice(cities))
                 for i in range(1, 1001)])

def sqoop_import(table, split_by, m, target):
    lo, hi = con.execute(f"SELECT MIN({split_by}), MAX({split_by}) FROM {table}").fetchone()
    print(f"BoundingValsQuery: MIN={lo}, MAX={hi}, mappers={m}")
    os.makedirs(target, exist_ok=True)
    step = (hi - lo + 1) / m
    for i in range(m):
        a, b = int(lo + i * step), int(lo + (i + 1) * step)
        where = f"{split_by} >= {a} AND {split_by} < {b}"
        rows = con.execute(f"SELECT * FROM {table} WHERE {where}").fetchall()
        with open(f"{target}/part-m-{i:05d}", "w") as f:
            f.writelines(",".join(map(str, r)) + "\n" for r in rows)
        print(f"  mapper {i}: WHERE {where:<22} -> {len(rows)} rows -> part-m-{i:05d}")

target = os.path.join(tempfile.mkdtemp(), "orders")      # stands in for --target-dir /data/orders
sqoop_import("orders", "id", 4, target)
print(sorted(os.listdir(target)))
print("first line of part-m-00000:", open(f"{target}/part-m-00000").readline().strip())
