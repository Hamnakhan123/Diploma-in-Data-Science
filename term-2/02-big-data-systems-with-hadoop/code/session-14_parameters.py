import argparse
from pyspark.sql import SparkSession, functions as F
spark = (SparkSession.builder.master("local[2]").appName("params")
         .config("spark.ui.showConsoleProgress", "false").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
sc = spark.sparkContext

# 1) job arguments (in production: spark-submit job.py --city Karachi --min-amount 3000)
parser = argparse.ArgumentParser()
parser.add_argument("--city", default="Karachi")
parser.add_argument("--min-amount", type=float, default=3000)
args = parser.parse_args([])                                   # [] so it also runs in a notebook

spark.createDataFrame([(1, "Karachi", 5200.0), (2, "Lahore", 900.0), (3, "Karachi", 15500.0),
                       (4, "Lahore", 7300.0), (5, "Karachi", 2100.0)],
                      "id int, city string, amount double").createOrReplaceTempView("sales")

# 2) named parameter markers: values are bound, never pasted into the SQL text
q = "SELECT id, city, amount FROM sales WHERE city = :city AND amount >= :min_amt ORDER BY id"
spark.sql(q, args={"city": args.city, "min_amt": args.min_amount}).show()

# 3) IDENTIFIER() for a table name chosen at run time
print("rows:", spark.sql("SELECT COUNT(*) AS n FROM IDENTIFIER(:tbl)", args={"tbl": "sales"}).first()["n"])

# 4) a malicious value is treated as data, not SQL
evil = "Karachi' OR '1'='1"
print("injection attempt rows:", spark.sql("SELECT * FROM sales WHERE city = :city", args={"city": evil}).count())

# 5) broadcast variable + accumulator on an RDD
tax = sc.broadcast({"Karachi": 0.13, "Lahore": 0.16})          # shipped once per executor
bad = sc.accumulator(0)
def add_tax(r):
    rate = tax.value.get(r[1])
    if rate is None:
        bad.add(1); return None
    return (r[0], r[1], round(r[2] * (1 + rate), 1))
rows = sc.parallelize([(1, "Karachi", 1000.0), (2, "Lahore", 1000.0), (3, "Quetta", 1000.0)])
print("with tax:", rows.map(add_tax).filter(lambda x: x).collect(), "| unknown cities:", bad.value)
