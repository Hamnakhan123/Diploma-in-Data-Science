import os, tempfile
from pyspark.sql import SparkSession, functions as F
base = tempfile.mkdtemp()
spark = (SparkSession.builder.master("local[2]").appName("buckets")
         .config("spark.sql.warehouse.dir", os.path.join(base, "warehouse"))
         .config("spark.sql.autoBroadcastJoinThreshold", "-1")
         .config("spark.ui.showConsoleProgress", "false").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")   # force a real join so bucketing matters
orders = spark.range(0, 20000).select((F.col("id") % 500).alias("customer_id"), (F.col("id") % 977).alias("amount"))
custs = spark.range(0, 500).select(F.col("id").alias("customer_id"), (F.col("id") % 5).alias("segment"))
orders.write.bucketBy(8, "customer_id").sortBy("customer_id").mode("overwrite").saveAsTable("orders_b")
custs.write.bucketBy(8, "customer_id").sortBy("customer_id").mode("overwrite").saveAsTable("customers_b")

wh = spark.conf.get("spark.sql.warehouse.dir").replace("file:", "")
files = [f for f in os.listdir(os.path.join(wh, "orders_b")) if f.endswith(".parquet")]
print("orders_b data files:", len(files))
j = spark.table("orders_b").join(spark.table("customers_b"), "customer_id")
plan = j._jdf.queryExecution().executedPlan().toString()
print("shuffle needed for join:", "Exchange hashpartitioning" in plan)
print("rows joined:", j.count())
spark.sql("SELECT segment, COUNT(*) AS orders, SUM(amount) AS revenue FROM orders_b JOIN customers_b USING (customer_id) GROUP BY segment ORDER BY segment").show()
