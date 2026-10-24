import sqlite3
con = sqlite3.connect(":memory:")
con.execute("CREATE TABLE orders(id INTEGER PRIMARY KEY, amount REAL)")
con.executemany("INSERT INTO orders VALUES (?,?)", [(i, i * 10.0) for i in range(1, 101)])
last_value = 0

def incremental_import():
    global last_value
    rows = con.execute("SELECT * FROM orders WHERE id > ? ORDER BY id", (last_value,)).fetchall()
    if rows:
        last_value = rows[-1][0]
    print(f"imported {len(rows):>3} new rows; saved --last-value = {last_value}")

incremental_import()                                   # night 1: full load
con.executemany("INSERT INTO orders VALUES (?,?)", [(i, i * 10.0) for i in range(101, 131)])
incremental_import()                                   # night 2: only the 30 new rows
incremental_import()                                   # night 3: nothing new
