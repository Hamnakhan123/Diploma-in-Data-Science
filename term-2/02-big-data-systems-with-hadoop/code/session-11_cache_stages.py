import time
from pyspark.sql import SparkSession, functions as F
spark = (SparkSession.builder.master("local[2]").appName("cache")
         .config("spark.sql.shuffle.partitions", "4")
         .config("spark.ui.showConsoleProgress", "false").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

big = (spark.range(0, 3_000_000)
       .withColumn("city", (F.col("id") % 5).cast("string"))
       .withColumn("amount", F.sqrt(F.col("id")) * 3))
clean = big.filter(F.col("amount") > 100).withColumn("tax", F.col("amount") * 0.17)

def timed(label, fn):
    t0 = time.time(); out = fn(); print(f"{label:<28} {time.time() - t0:6.2f} s  -> {out}")

timed("count, not cached", lambda: clean.count())
clean.cache()
timed("count, first (fills cache)", lambda: clean.count())
timed("count, from cache", lambda: clean.count())
timed("group by city, from cache", lambda: clean.groupBy("city").count().count())
print("storage level:", clean.storageLevel)
clean.unpersist()
