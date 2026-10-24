import math
from pyspark.sql import SparkSession
spark = (SparkSession.builder.master("local[2]").appName("tfidf")
         .config("spark.ui.showConsoleProgress", "false").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
sc = spark.sparkContext

docs = sc.parallelize([("d1", "big data big clusters"), ("d2", "big data in hdfs"),
                       ("d3", "spark streams data fast")])
N = docs.count()
tf = (docs.flatMap(lambda d: [((w, d[0]), 1) for w in d[1].split()])
          .reduceByKey(lambda a, b: a + b)                                   # job 1
          .map(lambda x: (x[0][1], (x[0][0], x[1]))))
doc_len = docs.map(lambda d: (d[0], len(d[1].split())))
tf = tf.join(doc_len).map(lambda x: (x[1][0][0], (x[0], x[1][0][1] / x[1][1])))  # job 2
df = tf.map(lambda x: (x[0], 1)).reduceByKey(lambda a, b: a + b)
tfidf = tf.join(df).map(lambda x: (x[1][0][0], x[0], round(x[1][0][1] * math.log(N / x[1][1]), 3)))  # job 3
for d, w, s in sorted(tfidf.collect(), key=lambda r: (r[0], -r[2], r[1])):
    print(d, f"{w:<9}", s)
