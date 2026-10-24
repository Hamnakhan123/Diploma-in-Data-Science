import os, tempfile
from pyspark.sql import SparkSession, functions as F
base = tempfile.mkdtemp()
spark = (SparkSession.builder.master("local[2]").appName("partitions")
         .config("spark.sql.warehouse.dir", os.path.join(base, "warehouse"))
         .config("spark.sql.shuffle.partitions", "4")
         .config("spark.ui.showConsoleProgress", "false").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

sales = (spark.range(0, 6000)
         .withColumn("year", F.when(F.col("id") % 2 == 0, 2025).otherwise(2026))
         .withColumn("month", (F.floor(F.col("id") / 2) % 12 + 1).cast("int"))
         .withColumn("city", F.element_at(F.array(*[F.lit(c) for c in ["Karachi", "Lahore", "Islamabad"]]),
                                          (F.floor(F.col("id") / 24) % 3 + 1).cast("int")))
         .withColumn("amount", (F.col("id") * 37 % 5000 + 100).cast("double")))

sales.write.partitionBy("year", "month").mode("overwrite").saveAsTable("sales")
spark.sql("SHOW PARTITIONS sales").show(5)
tdir = os.path.join(spark.conf.get("spark.sql.warehouse.dir").replace("file:", ""), "sales")
print("year folders:", sorted(d for d in os.listdir(tdir) if d.startswith("year")))
print("month folders in year=2026:", len(os.listdir(os.path.join(tdir, "year=2026"))))

q = spark.sql("SELECT city, SUM(amount) AS total FROM sales WHERE year = 2026 AND month = 10 GROUP BY city ORDER BY city")
q.show()
plan = q._jdf.queryExecution().executedPlan().toString()
import re
print(re.sub(r"#\d+", "", re.search(r"PartitionFilters: \[[^\]]*\]", plan).group(0)))
print("total partitions:", spark.sql("SHOW PARTITIONS sales").count(), "| folders scanned: year=2026/month=10 only")
