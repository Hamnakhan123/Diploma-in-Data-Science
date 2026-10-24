import os, tempfile
from pyspark.sql import SparkSession
base = tempfile.mkdtemp()
spark = (SparkSession.builder.master("local[2]").appName("hive-tables")
         .config("spark.sql.warehouse.dir", os.path.join(base, "warehouse"))
         .config("spark.ui.showConsoleProgress", "false").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

ext_dir = os.path.join(base, "landing", "cities")
spark.createDataFrame([("Karachi", "Sindh"), ("Lahore", "Punjab"), ("Quetta", "Balochistan")],
                      ["city", "province"]).write.mode("overwrite").parquet(ext_dir)

wh = spark.conf.get("spark.sql.warehouse.dir").replace("file:", "")
spark.sql("DROP TABLE IF EXISTS cities_managed"); spark.sql("DROP TABLE IF EXISTS cities_external")
spark.sql("CREATE TABLE cities_managed (city STRING, province STRING) USING parquet")
spark.sql("INSERT INTO cities_managed SELECT * FROM parquet.`%s`" % ext_dir)
spark.sql(f"CREATE TABLE cities_external USING parquet LOCATION '{ext_dir}'")
spark.sql("SELECT * FROM cities_external ORDER BY city").show()

for t in ["cities_managed", "cities_external"]:
    kind = spark.sql(f"DESCRIBE TABLE EXTENDED {t}").filter("col_name = 'Type'").first()["data_type"]
    print(f"{t:<16} type = {kind}")

managed_dir = os.path.join(wh, "cities_managed")
spark.sql("DROP TABLE cities_managed"); spark.sql("DROP TABLE cities_external")
print("\nafter DROP TABLE:")
print("  managed files still exist? ", os.path.exists(managed_dir))
print("  external files still exist?", os.path.exists(ext_dir))
