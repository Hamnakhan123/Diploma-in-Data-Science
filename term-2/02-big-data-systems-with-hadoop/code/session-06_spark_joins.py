from pyspark.sql import SparkSession, functions as F
spark = (SparkSession.builder.master("local[2]").appName("joins")
         .config("spark.ui.showConsoleProgress", "false").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

customers = spark.createDataFrame([(1, "Ayesha", "Karachi"), (2, "Bilal", "Lahore"), (3, "Sana", "Multan")],
                                  ["cust_id", "name", "city"])
orders = spark.createDataFrame([(101, 1, 2500), (102, 2, 900), (103, 1, 4100), (104, 4, 700)],
                               ["order_id", "cust_id", "amount"])

orders.join(customers, "cust_id", "inner").orderBy("order_id").show()
orders.join(customers, "cust_id", "left").orderBy("order_id").show()

def join_strategy(df):
    plan = df._jdf.queryExecution().executedPlan().toString()
    return "BroadcastHashJoin" if "BroadcastHashJoin" in plan else "SortMergeJoin" if "SortMergeJoin" in plan else plan[:60]

spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)       # disable automatic broadcast
print("default plan     :", join_strategy(orders.join(customers, "cust_id")))
print("with broadcast() :", join_strategy(orders.join(F.broadcast(customers), "cust_id")))
