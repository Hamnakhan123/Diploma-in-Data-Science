from pyspark.sql import SparkSession, functions as F
spark = (SparkSession.builder.master("local[2]").appName("stats")
         .config("spark.sql.shuffle.partitions", "4")
         .config("spark.ui.showConsoleProgress", "false").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

cities = F.array(*[F.lit(c) for c in ["Karachi", "Lahore", "Islamabad", "Faisalabad"]])
cats = F.array(*[F.lit(c) for c in ["Electronics", "Fashion", "Grocery"]])
orders = (spark.range(50000)
    .withColumn("city", F.element_at(cities, (F.abs(F.hash("id")) % 4 + 1).cast("int")))
    .withColumn("category", F.element_at(cats, (F.abs(F.hash(F.col("id") * 7)) % 3 + 1).cast("int")))
    .withColumn("items", (F.rand(seed=1) * 5 + 1).cast("int"))
    .withColumn("amount", F.round(F.exp(F.randn(seed=2) * 0.6 + 7.5) * F.col("items") / 2, 0))
    .withColumn("month", (F.col("id") % 12 + 1).cast("int")))

orders.select("items", "amount").summary("count", "mean", "stddev", "min", "25%", "50%", "75%", "max").show()
q = orders.approxQuantile("amount", [0.5, 0.9, 0.99], 0.001)
print("approx median, p90, p99 of amount:", q)
print("corr(items, amount) =", round(orders.stat.corr("items", "amount"), 3))
orders.agg(F.round(F.skewness("amount"), 2).alias("skewness"),
           F.approx_count_distinct("city").alias("distinct_cities")).show()
orders.stat.crosstab("city", "category").orderBy("city_category").show()
