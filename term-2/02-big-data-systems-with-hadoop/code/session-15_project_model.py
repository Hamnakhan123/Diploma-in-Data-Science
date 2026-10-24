import os, tempfile
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
spark = (SparkSession.builder.master("local[2]").appName("final-model")
         .config("spark.sql.shuffle.partitions", "4")
         .config("spark.ui.showConsoleProgress", "false").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

zones = F.array(*[F.lit(z) for z in ["Clifton", "Saddar", "Gulshan", "DHA", "Korangi", "North Nazimabad"]])
trips = (spark.range(60000)
    .withColumn("zone", F.element_at(zones, (F.abs(F.hash("id")) % 6 + 1).cast("int")))
    .withColumn("hour", (F.abs(F.hash(F.col("id") * 3)) % 24).cast("int"))
    .withColumn("km", F.round(F.exp(F.randn(seed=2) * 0.5 + 1.8), 1))
    .withColumn("fare", F.round(F.col("km") * 55 + 120 + F.when(F.col("hour").between(17, 20), 80).otherwise(0)
                                + F.randn(seed=3) * 30, 0)))

checks = trips.agg(F.sum(F.col("km").isNull().cast("int")).alias("null_km"),
                   F.sum((F.col("fare") <= 0).cast("int")).alias("non_positive_fares"),
                   F.min("km").alias("min_km"), F.max("km").alias("max_km"))
print("data-quality checks:", checks.first().asDict())

train, test = trips.randomSplit([0.8, 0.2], seed=42)       # not a time series, so a random split is fine
pipe = Pipeline(stages=[
    StringIndexer(inputCol="zone", outputCol="zone_idx"),
    OneHotEncoder(inputCols=["zone_idx"], outputCols=["zone_vec"]),
    VectorAssembler(inputCols=["km", "zone_vec"], outputCol="features"),
    LinearRegression(labelCol="fare")])
peak = F.col("hour").between(17, 20).cast("int")
model_plain = pipe.fit(train)
pipe_peak = Pipeline(stages=pipe.getStages()[:2] + [
    VectorAssembler(inputCols=["km", "zone_vec", "peak"], outputCol="features"), LinearRegression(labelCol="fare")])
model_peak = pipe_peak.fit(train.withColumn("peak", peak))

ev = RegressionEvaluator(labelCol="fare", metricName="rmse")
mean_fare = train.agg(F.avg("fare")).first()[0]
print(f"baseline RMSE (always predict mean fare): {ev.evaluate(test.withColumn('prediction', F.lit(mean_fare))):.1f}")
print(f"model RMSE (km + zone)                  : {ev.evaluate(model_plain.transform(test)):.1f}")
print(f"model RMSE (km + zone + peak hour)      : {ev.evaluate(model_peak.transform(test.withColumn('peak', peak))):.1f}")
lr = model_peak.stages[-1]
print(f"fare ~ {lr.intercept:.0f} + {lr.coefficients[0]:.1f} x km ... + {lr.coefficients[-1]:.0f} at peak hours")
