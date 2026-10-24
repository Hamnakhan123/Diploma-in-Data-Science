from pyspark.sql import SparkSession, functions as F
spark = (SparkSession.builder.master("local[2]").appName("pig-logs")
         .config("spark.sql.shuffle.partitions", "4")
         .config("spark.ui.showConsoleProgress", "false").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

logs = spark.createDataFrame([
    ("10.0.0.1", "10:01", "/home", 200, 5120), ("10.0.0.2", "10:02", "/admissions", 200, 8192),
    ("10.0.0.1", "10:03", "/results", 500, 0), ("10.0.0.3", "10:04", "/home", 200, 5120),
    ("10.0.0.1", "10:05", "/results", 500, 0), ("10.0.0.4", "10:06", "/fees", 404, 0),
    ("10.0.0.2", "10:07", "/fees", 200, 3072), ("10.0.0.1", "10:08", "/admissions", 503, 0),
    ("10.0.0.3", "10:09", "/results", 200, 9216), ("10.0.0.2", "10:10", "/home", 200, 5120)],
    "ip string, ts string, url string, status int, bytes long")
pages = spark.createDataFrame([("/home", "main"), ("/admissions", "students"), ("/results", "students"),
                               ("/fees", "finance")], "url string, section string")

errors = logs.filter("status >= 500")                     # SPLIT ... errors IF status >= 500
ok = logs.filter("status < 400")                          #           ok     IF status < 400
print("errors by url:")
errors.groupBy("url").agg(F.count("*").alias("n")).orderBy(F.desc("n")).show()

print("traffic by section (replicated join = broadcast):")
(ok.join(F.broadcast(pages), "url")
   .groupBy("section").agg(F.count("*").alias("hits"), F.round(F.sum("bytes") / 1024.0, 1).alias("kb"))
   .orderBy(F.desc("hits"), "section").show())

print("top 3 IPs:")
logs.groupBy("ip").agg(F.count("*").alias("n")).orderBy(F.desc("n"), "ip").limit(3).show()

print("Pig-style bag per IP (collect_list = bag):")
logs.groupBy("ip").agg(F.collect_list(F.struct("url", "status")).alias("bag")).orderBy("ip").show(truncate=False)
