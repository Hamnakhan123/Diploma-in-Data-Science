import time
from pyspark.sql import SparkSession
spark = (SparkSession.builder.master("local[2]").appName("lazy")
         .config("spark.ui.showConsoleProgress", "false").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
sc = spark.sparkContext

t0 = time.time()
rdd = sc.parallelize(range(1, 2_000_001), 4)
squares = rdd.map(lambda x: x * x).filter(lambda x: x % 3 == 0)    # transformations only
print(f"after transformations : {time.time() - t0:.3f} s (nothing has run yet)")
print("count =", squares.count())                                   # action -> job runs now
print("the action triggered a job: map + filter ran over all 4 partitions")
print("first 5 =", squares.take(5))
