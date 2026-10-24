from pyspark.sql import SparkSession, functions as F
spark = (SparkSession.builder.master("local[2]").appName("json")
         .config("spark.ui.showConsoleProgress", "false").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

events = spark.createDataFrame([
  ('{"user":{"id":7,"city":"Karachi"},"items":[{"sku":"BK-1","qty":2,"price":450.0},{"sku":"PN-9","qty":1,"price":120.0}],"tags":{"channel":"app","promo":"SALE1111"}}',),
  ('{"user":{"id":9,"city":"Lahore"},"items":[{"sku":"BK-1","qty":1,"price":450.0}],"tags":{"channel":"web"}}',)], ["raw"])

schema = ("user STRUCT<id: INT, city: STRING>, "
          "items ARRAY<STRUCT<sku: STRING, qty: INT, price: DOUBLE>>, tags MAP<STRING, STRING>")
parsed = events.select(F.from_json("raw", schema).alias("e")).select("e.*")
parsed.printSchema()

parsed.select(F.col("user.id").alias("user_id"), "user.city",
              F.size("items").alias("n_items"),
              F.aggregate("items", F.lit(0.0), lambda acc, x: acc + x.qty * x.price).alias("basket_value"),
              F.element_at("tags", "channel").alias("channel"),
              F.coalesce(F.element_at("tags", "promo"), F.lit("none")).alias("promo")).show()

parsed.select(F.col("user.id").alias("user_id"), F.explode("items").alias("item")) \
      .select("user_id", "item.sku", "item.qty", "item.price").show()
