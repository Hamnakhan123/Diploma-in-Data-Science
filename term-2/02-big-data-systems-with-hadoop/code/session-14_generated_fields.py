from pyspark.sql import SparkSession, functions as F, Window
from pyspark.sql.functions import pandas_udf
import pandas as pd
spark = (SparkSession.builder.master("local[2]").appName("generated")
         .config("spark.sql.shuffle.partitions", "2")
         .config("spark.ui.showConsoleProgress", "false").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

sales = spark.createDataFrame([("Karachi", "Ayesha", 5200.0), ("Lahore", "Bilal", 900.0),
                               ("Karachi", "Omar", 15500.0), ("Lahore", "Sana", 7300.0),
                               ("Karachi", "Hina", 2100.0)], "city string, customer string, amount double")

@pandas_udf("string")
def initials(names: pd.Series) -> pd.Series:          # vectorised UDF: runs on whole batches
    return names.str[0] + "."

w_city = Window.partitionBy("city")
gen = (sales
   .withColumn("row_id", F.row_number().over(Window.orderBy("customer")))
   .withColumn("tier", F.when(F.col("amount") >= 10000, "Gold").when(F.col("amount") >= 5000, "Silver").otherwise("Bronze"))
   .withColumn("gst", F.round(F.col("amount") * 0.18, 2))
   .withColumn("share_of_city_%", F.round(F.col("amount") / F.sum("amount").over(w_city) * 100, 1))
   .withColumn("rank_in_city", F.rank().over(w_city.orderBy(F.desc("amount"))))
   .withColumn("cust_key", F.substring(F.sha2(F.col("customer"), 256), 1, 10))
   .withColumn("anon", initials("customer")))
gen.orderBy("row_id").show()
