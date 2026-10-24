from pyspark.sql import SparkSession, functions as F
spark = (SparkSession.builder.master("local[2]").appName("kpis")
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

kpi = orders.agg(F.count("*").alias("orders"), F.sum("amount").alias("revenue"),
                 F.round(F.avg("amount"), 0).alias("avg_order")).first()
print(f"KPI cards -> orders: {kpi['orders']:,} | revenue: Rs {kpi['revenue']:,.0f} | avg order: Rs {kpi['avg_order']:,.0f}\n")

print("rollup: revenue by city with grand total (in Rs million)")
(orders.rollup("city").agg(F.round(F.sum("amount") / 1e6, 2).alias("revenue_m"))
       .orderBy(F.col("city").isNull(), "city").show())

print("pivot: revenue (Rs million) by category and quarter")
(orders.withColumn("quarter", F.concat(F.lit("Q"), F.ceil(F.col("month") / 3).cast("string")))
       .groupBy("category").pivot("quarter").agg(F.round(F.sum("amount") / 1e6, 2))
       .orderBy("category").show())
