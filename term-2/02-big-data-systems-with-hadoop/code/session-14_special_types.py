from decimal import Decimal
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StructType, StructField, StringType, DecimalType
spark = (SparkSession.builder.master("local[2]").appName("types")
         .config("spark.sql.session.timeZone", "Asia/Karachi")
         .config("spark.ui.showConsoleProgress", "false").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
spark.conf.set("spark.sql.session.timeZone", "Asia/Karachi")

schema = StructType([StructField("order_id", StringType()), StructField("placed", StringType()),
                     StructField("price", DecimalType(10, 2)), StructField("phone", StringType())])
df = spark.createDataFrame([("A1", "24/10/2026 21:15", Decimal("0.10"), "03001234567"),
                            ("A2", "01/11/2026 09:05", Decimal("0.20"), "03217654321")], schema)
typed = (df.withColumn("placed_ts", F.to_timestamp("placed", "dd/MM/yyyy HH:mm"))
           .withColumn("placed_date", F.to_date("placed_ts"))
           .withColumn("weekday", F.date_format("placed_ts", "EEEE"))
           .withColumn("utc", F.to_utc_timestamp("placed_ts", "Asia/Karachi"))
           .withColumn("days_to_eom", F.datediff(F.last_day("placed_date"), F.col("placed_date"))))
typed.select("order_id", "placed_ts", "utc", "weekday", "days_to_eom", "phone").show(truncate=False)
typed.printSchema()

print("decimal sum :", df.agg(F.sum("price")).first()[0])
print("double sum  :", 0.1 + 0.2)
