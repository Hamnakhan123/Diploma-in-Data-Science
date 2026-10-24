import math
from pyspark.sql import SparkSession, functions as F, Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
spark = (SparkSession.builder.master("local[2]").appName("forecast")
         .config("spark.sql.shuffle.partitions", "4")
         .config("spark.ui.showConsoleProgress", "false").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

# 2 years of daily sales: trend + weekly pattern (Friday and weekend peaks) + noise
days = (spark.range(730).withColumnRenamed("id", "t")
        .withColumn("date", F.date_add(F.lit("2024-11-01").cast("date"), F.col("t").cast("int")))
        .withColumn("dow", F.dayofweek("date"))
        .withColumn("sales", F.round(200 + 0.15 * F.col("t")
                     + F.when(F.col("dow").isin(6, 7, 1), 60).otherwise(0)
                     + 20 * F.sin(2 * math.pi * F.col("t") / 365) + F.randn(seed=7) * 12, 1)))

w = Window.orderBy("t")
feat = (days.withColumn("lag_1", F.lag("sales", 1).over(w))
            .withColumn("lag_7", F.lag("sales", 7).over(w))
            .withColumn("roll_7", F.avg("sales").over(w.rowsBetween(-7, -1)))
            .withColumn("weekend", F.col("dow").isin(6, 7, 1).cast("int"))
            .na.drop())
cut = "2026-08-01"
train, test = feat.filter(F.col("date") < cut), feat.filter(F.col("date") >= cut)
print(f"train days: {train.count()}, test days: {test.count()} (test starts {cut})")

va = VectorAssembler(inputCols=["t", "lag_1", "lag_7", "roll_7", "weekend"], outputCol="features")
tr, te = va.transform(train), va.transform(test)
models = {"Linear regression": LinearRegression(labelCol="sales"),
          "Random forest": RandomForestRegressor(labelCol="sales", numTrees=50, seed=1)}

def metrics(df, pred):
    r = {m: RegressionEvaluator(labelCol="sales", predictionCol=pred, metricName=m).evaluate(df) for m in ["mae", "rmse"]}
    r["mape"] = df.select(F.avg(F.abs((F.col("sales") - F.col(pred)) / F.col("sales"))) * 100).first()[0]
    return r

print(f"\n{'model':<26}{'MAE':>7}{'RMSE':>7}{'MAPE %':>8}")
base = te.withColumn("naive", F.col("lag_7"))
r = metrics(base, "naive"); print(f"{'Naive (same day last wk)':<26}{r['mae']:>7.1f}{r['rmse']:>7.1f}{r['mape']:>8.1f}")
for name, m in models.items():
    p = m.fit(tr).transform(te)
    r = metrics(p, "prediction"); print(f"{name:<26}{r['mae']:>7.1f}{r['rmse']:>7.1f}{r['mape']:>8.1f}")

lr = models["Linear regression"].fit(tr)
print("\nlast 5 test days:")
lr.transform(te).select("date", "sales", F.round("prediction", 1).alias("forecast")).orderBy(F.desc("date")).limit(5).orderBy("date").show()
