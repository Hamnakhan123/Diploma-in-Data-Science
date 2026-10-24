from pyspark.sql import SparkSession, functions as F, Window
spark = (SparkSession.builder.master("local[2]").appName("windows")
         .config("spark.sql.shuffle.partitions", "4")
         .config("spark.ui.showConsoleProgress", "false").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

monthly = (spark.range(1, 13).withColumnRenamed("id", "month")
           .withColumn("revenue", F.round(10 + F.col("month") * 0.8 + F.sin(F.col("month")) * 1.5, 2)))  # Rs million
w = Window.orderBy("month")
trend = (monthly
    .withColumn("prev", F.lag("revenue").over(w))
    .withColumn("mom_growth_%", F.round((F.col("revenue") / F.col("prev") - 1) * 100, 1))
    .withColumn("moving_avg_3", F.round(F.avg("revenue").over(w.rowsBetween(-2, 0)), 2))
    .withColumn("running_total", F.round(F.sum("revenue").over(w.rowsBetween(Window.unboundedPreceding, 0)), 2))
    .drop("prev"))
trend.show()
