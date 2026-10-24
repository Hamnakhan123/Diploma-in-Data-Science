from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
spark = (SparkSession.builder.master("local[2]").appName("kmeans")
         .config("spark.sql.shuffle.partitions", "4")
         .config("spark.ui.showConsoleProgress", "false").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

seg = F.when(F.col("customer_id") % 10 < 5, 0).when(F.col("customer_id") % 10 < 8, 1).otherwise(2)
orders = (spark.range(60000)
    .withColumn("customer_id", (F.col("id") % 1500).cast("int")).withColumn("seg", seg)
    .withColumn("amount", F.round(F.when(F.col("seg") == 0, 800).when(F.col("seg") == 1, 2500).otherwise(9000)
                                  * F.exp(F.randn(seed=4) * 0.3), 0))
    .filter((F.col("seg") != 0) | (F.rand(seed=5) < 0.6)))
cust = orders.groupBy("customer_id").agg(F.count("*").alias("orders"),
                                         F.avg("amount").alias("avg_order")).cache()

assembler = VectorAssembler(inputCols=["orders", "avg_order"], outputCol="raw")
scaler = StandardScaler(inputCol="raw", outputCol="features", withMean=True, withStd=True)
evaluator = ClusteringEvaluator(featuresCol="features", metricName="silhouette")

print(" k  silhouette")
for k in [2, 3, 4, 5, 6]:
    model = Pipeline(stages=[assembler, scaler, KMeans(k=k, seed=42, featuresCol="features")]).fit(cust)
    print(f"{k:>2}  {evaluator.evaluate(model.transform(cust)):.3f}")

best = Pipeline(stages=[assembler, scaler, KMeans(k=3, seed=42, featuresCol="features")]).fit(cust)
profiles = (best.transform(cust).groupBy("prediction")
            .agg(F.count("*").alias("customers"), F.round(F.avg("orders"), 1).alias("avg_orders"),
                 F.round(F.avg("avg_order"), 0).alias("avg_order_value"))
            .orderBy("avg_order_value"))
profiles.show()
