from pyspark.sql import SparkSession, functions as F
spark = (SparkSession.builder.master("local[2]").appName("df")
         .config("spark.sql.shuffle.partitions", "4")
         .config("spark.ui.showConsoleProgress", "false").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

df = spark.createDataFrame([
    ("Ali", "CS", 3.4, 2023), ("Sara", "DS", 3.9, 2022), ("Omar", "CS", 2.8, 2023),
    ("Hina", "DS", 3.6, 2023), ("Zain", "SE", 3.1, 2022), ("Iqra", "CS", 3.7, 2022)],
    "name string, dept string, cgpa double, batch int")
df.printSchema()

result = (df.filter(F.col("cgpa") >= 3.0)
            .groupBy("dept").agg(F.count("*").alias("students"), F.round(F.avg("cgpa"), 2).alias("avg_cgpa"))
            .orderBy("dept"))
result.show()

df.createOrReplaceTempView("students")
spark.sql("SELECT batch, MAX(cgpa) AS best FROM students GROUP BY batch ORDER BY batch").show()
result.explain()
