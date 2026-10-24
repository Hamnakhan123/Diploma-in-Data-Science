import os, tempfile
from pyspark.sql import SparkSession, functions as F
spark = (SparkSession.builder.master("local[2]").appName("final-project")
         .config("spark.sql.shuffle.partitions", "4")
         .config("spark.sql.session.timeZone", "Asia/Karachi")
         .config("spark.ui.showConsoleProgress", "false").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
spark.conf.set("spark.sql.session.timeZone", "Asia/Karachi")
PARAMS = {"base": tempfile.mkdtemp(), "start": "2026-09-01", "days": 30, "fare_per_km": 55.0}

# ---------- BRONZE: ingest raw data as-is ----------
zones = F.array(*[F.lit(z) for z in ["Clifton", "Saddar", "Gulshan", "DHA", "Korangi", "North Nazimabad"]])
raw = (spark.range(120000)
    .withColumn("pickup_ts", (F.unix_timestamp(F.lit(PARAMS["start"]).cast("timestamp"))
                              + (F.rand(seed=1) * PARAMS["days"] * 86400)).cast("timestamp"))
    .withColumn("zone", F.element_at(zones, (F.abs(F.hash("id")) % 6 + 1).cast("int")))
    .withColumn("km", F.round(F.exp(F.randn(seed=2) * 0.5 + 1.8), 1))
    .withColumn("fare", F.round(F.col("km") * PARAMS["fare_per_km"] + 120 + F.randn(seed=3) * 30, 0))
    .withColumn("km", F.when(F.rand(seed=4) < 0.005, -1.0).otherwise(F.col("km"))))     # bad records
raw = raw.unionByName(raw.limit(300))                                                     # duplicates
bronze = os.path.join(PARAMS["base"], "bronze", "trips")
raw.write.mode("overwrite").parquet(bronze)

# ---------- SILVER: clean, validate, partition ----------
b = spark.read.parquet(bronze)
silver_df = (b.dropDuplicates(["id"]).filter((F.col("km") > 0) & (F.col("fare") > 0))
              .withColumn("trip_date", F.to_date("pickup_ts"))
              .withColumn("hour", F.hour("pickup_ts")))
silver = os.path.join(PARAMS["base"], "silver", "trips")
silver_df.write.mode("overwrite").partitionBy("trip_date").parquet(silver)
n_b, n_s = b.count(), spark.read.parquet(silver).count()
print(f"bronze rows {n_b:,} -> silver rows {n_s:,} (removed {n_b - n_s:,} duplicates and invalid trips)")
print("silver date partitions:", len([d for d in os.listdir(silver) if d.startswith("trip_date=")]))

# ---------- GOLD: KPI tables for the dashboard ----------
s = spark.read.parquet(silver)
gold_zone = (s.groupBy("zone").agg(F.count("*").alias("trips"), F.round(F.avg("km"), 2).alias("avg_km"),
                                   F.round(F.sum("fare") / 1e6, 2).alias("revenue_m"))
              .orderBy(F.desc("trips")))
gold_zone.write.mode("overwrite").parquet(os.path.join(PARAMS["base"], "gold", "zone_kpis"))
gold_zone.show()
s.groupBy("hour").count().orderBy(F.desc("count")).limit(3).show()
