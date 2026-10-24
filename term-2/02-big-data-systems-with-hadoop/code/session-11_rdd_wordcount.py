from pyspark.sql import SparkSession
spark = (SparkSession.builder.master("local[2]").appName("rdd")
         .config("spark.ui.showConsoleProgress", "false").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
sc = spark.sparkContext

lines = sc.parallelize([
    "spark keeps data in memory", "hadoop writes data to disk",
    "spark runs on yarn", "yarn manages the hadoop cluster"], numSlices=2)
counts = (lines.flatMap(lambda l: l.split())       # narrow
               .map(lambda w: (w, 1))              # narrow
               .reduceByKey(lambda a, b: a + b))   # wide -> shuffle
print("partitions:", lines.getNumPartitions())
print("top words :", counts.sortBy(lambda x: (-x[1], x[0])).take(5))
import re
print("\nlineage:")
print(re.sub(r" at \S+:\d+ \[\]", "", counts.toDebugString().decode()))
