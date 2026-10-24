from pyspark.sql import SparkSession, functions as F, Window
spark = (SparkSession.builder.master("local[2]").appName("nested")
         .config("spark.ui.showConsoleProgress", "false").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
students = spark.createDataFrame([("Ali", "Karachi", 82), ("Sara", "Lahore", 91), ("Omar", "Karachi", 77),
                                  ("Hina", "Lahore", 88), ("Zain", "Karachi", 95), ("Iqra", "Lahore", 73)],
                                 "name string, city string, score int")
w = Window.partitionBy("city").orderBy(F.desc("score"))
(students.withColumn("pos", F.row_number().over(w)).filter("pos <= 2")    # ORDER + LIMIT inside each bag
         .groupBy("city").agg(F.collect_list("name").alias("top2")).orderBy("city").show())
