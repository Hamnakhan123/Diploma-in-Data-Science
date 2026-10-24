from pyspark.sql import SparkSession, functions as F
spark = (SparkSession.builder.master("local[2]").appName("pig-style")
         .config("spark.ui.showConsoleProgress", "false").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

sales = spark.createDataFrame(
    [(1, "Karachi", 2500.0), (2, "Lahore", 800.0), (3, "Karachi", 1200.0), (4, "Islamabad", 3100.0),
     (5, "Lahore", 1500.0), (6, "Karachi", 400.0), (7, "Lahore", 2600.0)],
    "order_id int, city string, amount double")                         # LOAD ... AS (...)
big = sales.filter(F.col("amount") > 1000)                               # FILTER
by_city = big.groupBy("city")                                            # GROUP
totals = by_city.agg(F.count("*").alias("orders"), F.sum("amount").alias("revenue"))  # FOREACH GENERATE
ranked = totals.orderBy(F.desc("revenue"))                               # ORDER
ranked.show()                                                            # DUMP
ranked.explain()                                                         # EXPLAIN
