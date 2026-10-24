from pyspark.sql import SparkSession
spark = (SparkSession.builder.master("local[2]").appName("hiveql")
         .config("spark.ui.showConsoleProgress", "false").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

spark.createDataFrame(
    [("Ali", "Karachi", ["hadoop", "spark"], 82), ("Sara", "Lahore", ["hive"], 91),
     ("Omar", "Karachi", ["spark", "hive", "pig"], 77), ("Hina", "Lahore", ["hadoop"], 88),
     ("Zain", "Karachi", ["pig"], 95)],
    "name string, city string, skills array<string>, score int").createOrReplaceTempView("students")

spark.sql("""
  SELECT skill, COUNT(*) AS students
  FROM students LATERAL VIEW explode(skills) s AS skill
  GROUP BY skill ORDER BY students DESC, skill
""").show()

spark.sql("""
  SELECT city, name, score,
         RANK() OVER (PARTITION BY city ORDER BY score DESC) AS rank_in_city,
         ROUND(AVG(score) OVER (PARTITION BY city), 1)     AS city_avg
  FROM students ORDER BY city, rank_in_city
""").show()
