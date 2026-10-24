from pyspark.sql import SparkSession, functions as F
spark = (SparkSession.builder.master("local[2]").appName("prep")
         .config("spark.sql.shuffle.partitions", "4")
         .config("spark.ui.showConsoleProgress", "false").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

# 3 hidden customer types: bargain hunters, regulars, premium buyers
seg = F.when(F.col("customer_id") % 10 < 5, 0).when(F.col("customer_id") % 10 < 8, 1).otherwise(2)
orders = (spark.range(60000)
    .withColumn("customer_id", (F.col("id") % 1500).cast("int"))
    .withColumn("seg", seg)
    .withColumn("amount", F.round(F.when(F.col("seg") == 0, 800).when(F.col("seg") == 1, 2500).otherwise(9000)
                                  * F.exp(F.randn(seed=4) * 0.3), 0))
    .filter((F.col("seg") != 0) | (F.rand(seed=5) < 0.6))        # bargain hunters order less often
    .withColumn("amount", F.when(F.rand(seed=6) < 0.01, None).otherwise(F.col("amount"))))  # 1% missing

print("rows:", orders.count(), "| null amounts:", orders.filter(F.col("amount").isNull()).count())
clean = orders.na.drop(subset=["amount"]).dropDuplicates(["id"])
cust = (clean.groupBy("customer_id").agg(
            F.count("*").alias("orders"),
            F.round(F.avg("amount"), 0).alias("avg_order"),
            F.round(F.sum("amount"), 0).alias("total_spend")))
cust.summary("count", "mean", "min", "max").show()
cust.orderBy("customer_id").show(5)
