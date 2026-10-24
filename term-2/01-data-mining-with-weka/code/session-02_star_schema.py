import sqlite3

con = sqlite3.connect(":memory:")
cur = con.cursor()
cur.executescript("""
CREATE TABLE dim_date(date_key INTEGER PRIMARY KEY, month TEXT, year INTEGER);
CREATE TABLE dim_product(product_key INTEGER PRIMARY KEY, name TEXT, category TEXT);
CREATE TABLE dim_store(store_key INTEGER PRIMARY KEY, city TEXT);
CREATE TABLE fact_sales(date_key INT, product_key INT, store_key INT, quantity INT, amount REAL);

INSERT INTO dim_date VALUES (1,'Jan',2025),(2,'Feb',2025),(3,'Jan',2026);
INSERT INTO dim_product VALUES (1,'Notebook','Stationery'),(2,'Pen','Stationery'),(3,'Headphones','Electronics');
INSERT INTO dim_store VALUES (1,'Karachi'),(2,'Lahore');
INSERT INTO fact_sales VALUES
 (1,1,1,40,4000),(1,2,1,100,2000),(1,3,2,5,25000),
 (2,1,2,30,3000),(2,3,1,8,40000),(3,2,2,150,3000),(3,3,1,6,30000);
""")

print("Roll-up: total sales by year and category")
for row in cur.execute("""
    SELECT d.year, p.category, SUM(f.amount) AS total
    FROM fact_sales f
    JOIN dim_date d ON f.date_key = d.date_key
    JOIN dim_product p ON f.product_key = p.product_key
    GROUP BY d.year, p.category ORDER BY d.year, total DESC"""):
    print(row)

print("\nSlice: 2025 only, sales by city")
for row in cur.execute("""
    SELECT s.city, SUM(f.amount), SUM(f.quantity)
    FROM fact_sales f
    JOIN dim_date d ON f.date_key = d.date_key
    JOIN dim_store s ON f.store_key = s.store_key
    WHERE d.year = 2025 GROUP BY s.city"""):
    print(row)
