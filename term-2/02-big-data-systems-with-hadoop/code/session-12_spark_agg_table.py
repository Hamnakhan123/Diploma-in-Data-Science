import os, tempfile
from pyspark.sql import SparkSession, functions as F
spark = (SparkSession.builder.master("local[2]").appName("agg-table")
         .config("spark.sql.shuffle.partitions", "4")
         .config("spark.ui.showConsoleProgress", "false").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
cities = F.array(*[F.lit(c) for c in ["Karachi", "Lahore", "Islamabad", "Faisalabad"]])
cats = F.array(*[F.lit(c) for c in ["Electronics", "Fashion", "Grocery"]])
orders = (spark.range(50000)
    .withColumn("city", F.element_at(cities, (F.abs(F.hash("id")) % 4 + 1).cast("int")))
    .withColumn("category", F.element_at(cats, (F.abs(F.hash(F.col("id") * 7)) % 3 + 1).cast("int")))
    .withColumn("amount", F.round(F.exp(F.randn(seed=2) * 0.6 + 7.5), 0))
    .withColumn("month", (F.col("id") % 12 + 1).cast("int")))

agg = (orders.groupBy("month", "city", "category")
       .agg(F.count("*").alias("orders"), F.sum("amount").alias("revenue")))
out = os.path.join(tempfile.mkdtemp(), "sales_agg")    # use out = "sales_agg" to keep it for the dashboard app
agg.write.mode("overwrite").partitionBy("month").parquet(out)

raw_rows, agg_rows = orders.count(), agg.count()
print(f"raw rows: {raw_rows:,} -> dashboard rows: {agg_rows:,} ({raw_rows // agg_rows}x smaller)")
print("month partitions written:", len([d for d in os.listdir(out) if d.startswith("month=")]))
pdf = spark.read.parquet(out).toPandas()            # small enough for pandas / a dashboard
print(pdf.groupby("city")["revenue"].sum().sort_values(ascending=False).round(0).to_string())
