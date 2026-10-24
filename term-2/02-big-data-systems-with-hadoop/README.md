# Module 2 · Foundations of Big Data Systems with Hadoop

> 📍 [Diploma in Data Science](../../README.md) › [Term 2](../README.md) › **Foundations of Big Data Systems with Hadoop**

⏱ **40 hours** · 15 sessions · Prepared by **Hamna Zahid Khan**, Junior Lecturer

[⬅ Data Mining with WEKA](../01-data-mining-with-weka/README.md) · [▶ Start Session 1](session-01-big-data-and-hadoop-ecosystem.md) · [Advanced Data Science with Python ➡](../03-advanced-data-science-with-python/README.md)

---

## 🎯 About this module

This module explains how organisations store and process data that is too big, too fast or too varied for one computer. You start with the ideas behind **Hadoop** (HDFS, YARN and MapReduce), learn to configure a cluster, move data with **Sqoop**, query it with **Hive** and **Impala**, script pipelines in **Pig**, and then move to **Apache Spark** for fast in-memory analysis, statistics, dashboards, clustering and forecasting. The module ends with a full project in the bronze → silver → gold pattern.

Hadoop needs a cluster, so every Hadoop concept is reproduced with a small **Python simulation** that shows exactly what the system does (block placement, failover, schedulers, MapReduce jobs, Sqoop splits). From Session 5 onwards the examples are **real PySpark** code that runs on your laptop or in Google Colab. Commands for a real cluster (`hdfs`, `yarn`, `sqoop`, `pig`, `hive`, `impala-shell`, `spark-submit`) are included and marked *illustrative*.

**By the end you will be able to:**

- explain Big Data with the 5 Vs and describe every major Hadoop ecosystem component;
- configure and size a Hadoop cluster and explain how HDFS stores, replicates and recovers data;
- explain YARN's architecture and choose a scheduler;
- design MapReduce algorithms for search, sorting, indexing, TF-IDF and joins;
- import relational data with Sqoop and query HDFS data with Hive and Impala using partitions and buckets;
- write Pig Latin data flows and translate them to Spark;
- analyse large data sets with Spark DataFrames, SQL and MLlib, and feed interactive dashboards;
- build a reproducible, parameterised Big Data pipeline for a final project.

## 📚 Sessions

| # | Session |
|---|---|
| 1 | [Introduction to Big Data and the Hadoop Ecosystem](session-01-big-data-and-hadoop-ecosystem.md) |
| 2 | [Configuring Hadoop and Setting Up a Cluster](session-02-configuring-hadoop-cluster.md) |
| 3 | [Understanding HDFS and Its Goals](session-03-hdfs-and-its-goals.md) |
| 4 | [Exploring YARN and Classic MapReduce](session-04-yarn-and-classic-mapreduce.md) |
| 5 | [Demonstrating MapReduce Algorithms](session-05-mapreduce-algorithms.md) |
| 6 | [Understanding Sqoop and Joins in MapReduce](session-06-sqoop-and-mapreduce-joins.md) |
| 7 | [Exploring Hadoop Architecture and Components](session-07-hadoop-architecture-and-components.md) |
| 8 | [Working with Hive and Impala](session-08-hive-and-impala.md) |
| 9 | [Introduction to Apache Pig](session-09-introduction-to-apache-pig.md) |
| 10 | [Pig Latin Data Model and Operators](session-10-pig-latin-data-model-and-operators.md) |
| 11 | [Introduction to Apache Spark](session-11-introduction-to-apache-spark.md) |
| 12 | [Spark Statistics for Interactive Dashboards](session-12-spark-statistics-and-dashboards.md) |
| 13 | [Data Analysis with Spark: Clustering and Forecasting](session-13-spark-clustering-and-forecasting.md) |
| 14 | [Special Field Types, Generated Fields and Parameters in Spark](session-14-spark-field-types-and-parameters.md) |
| 15 | [Final Project and Next Steps](session-15-final-project-and-next-steps.md) |
| 📝 | [Module practice test](practice-test.md) |

## 🛠️ Setup

1. **Python 3.10+** and **Java 17** (or 11). Check with `python --version` and `java -version`.
2. `pip install pyspark pandas numpy pyarrow` (add `streamlit plotly` for the Session 12 dashboard).
3. Run any script in [`code/`](code/) with `python session-XX_name.py`, or open a session notebook in **Google Colab** using the badge at the top of each session (Colab already has Java; the first cell installs PySpark).
4. **Optional, for the illustrative commands:** a single-node Hadoop 3.3 on Ubuntu or WSL2 (Session 2, Example 2), or a free cloud sandbox such as a Databricks Community workspace or a Google Dataproc trial.

All data in the examples is generated inside the code with fixed seeds, so every result in the notes can be reproduced exactly.

## 📋 Cheat sheet

### Hadoop daemons and ports (Hadoop 3)

| Daemon | Role | Web UI |
|---|---|---|
| NameNode | HDFS metadata (namespace, block map) | 9870 |
| DataNode | Stores blocks, sends heartbeats | 9864 |
| Secondary / Standby NameNode | Checkpoints fsimage + edits / HA failover | 9868 |
| ResourceManager | Cluster resources, scheduler | 8088 |
| NodeManager | Launches containers on a worker | 8042 |
| JobHistory Server | Finished MapReduce jobs | 19888 |
| Spark application / History Server | Jobs, stages, storage | 4040 / 18080 |

### HDFS and YARN commands

| Task | Command |
|---|---|
| List / make directory | `hdfs dfs -ls -h /data` · `hdfs dfs -mkdir -p /data/in` |
| Upload / download | `hdfs dfs -put local.csv /data/in` · `hdfs dfs -get /data/out/part-r-00000 .` |
| Read / space used | `hdfs dfs -cat file \| head` · `hdfs dfs -du -s -h /data` |
| Change replication | `hdfs dfs -setrep -w 2 /data/file` |
| Block health | `hdfs fsck /data -files -blocks -locations` |
| Cluster report | `hdfs dfsadmin -report` |
| HA state | `hdfs haadmin -getServiceState nn1` |
| Running apps / logs / kill | `yarn application -list` · `yarn logs -applicationId <id>` · `yarn application -kill <id>` |
| Run a MapReduce JAR | `hadoop jar app.jar MainClass /in /out` |

### Key configuration properties

| File | Property | Typical value |
|---|---|---|
| core-site.xml | `fs.defaultFS` | `hdfs://master:9000` |
| hdfs-site.xml | `dfs.replication` · `dfs.blocksize` | 3 · 134217728 (128 MB) |
| hdfs-site.xml | `dfs.namenode.name.dir` · `dfs.datanode.data.dir` | metadata dirs · one dir per disk |
| mapred-site.xml | `mapreduce.framework.name` | `yarn` |
| yarn-site.xml | `yarn.resourcemanager.hostname` · `yarn.nodemanager.aux-services` | `master` · `mapreduce_shuffle` |

### Sqoop, Hive and Pig

| Tool | Pattern |
|---|---|
| Sqoop import | `sqoop import --connect jdbc:mysql://db/shop --table orders --target-dir /data/orders -m 4 --split-by id` |
| Sqoop incremental | `... --incremental append --check-column id --last-value 5000` |
| Sqoop export | `sqoop export --connect ... --table daily_sales --export-dir /out/daily_sales` |
| Hive external table | `CREATE EXTERNAL TABLE t (...) ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' LOCATION '/path';` |
| Hive partitioned + bucketed | `CREATE TABLE s (...) PARTITIONED BY (year INT) CLUSTERED BY (id) INTO 16 BUCKETS STORED AS ORC;` |
| Impala refresh | `INVALIDATE METADATA t;` · `COMPUTE STATS t;` |
| Pig flow | `a = LOAD 'x' USING PigStorage(',') AS (...); b = FILTER a BY ...; g = GROUP b BY k; r = FOREACH g GENERATE group, COUNT(b);` |

### MapReduce patterns

| Problem | Map emits | Reduce does |
|---|---|---|
| Word count | `(word, 1)` | sum |
| Search (grep) | matching line | none (map-only) |
| Total-order sort | `(key, record)` + range partitioner | identity |
| Inverted index | `(word, doc_id)` | sorted unique doc list |
| Reduce-side join | `(key, tagged record)` from both tables | combine tags |
| Map-side join | joined record (small table cached) | none |

### Spark quick reference

| Task | PySpark |
|---|---|
| Session | `SparkSession.builder.master("local[*]").getOrCreate()` |
| Read / write | `spark.read.schema(s).csv(p, header=True)` · `df.write.partitionBy("date").parquet(p)` |
| Transform | `select`, `filter`, `withColumn`, `groupBy().agg()`, `join(F.broadcast(small), "k")` |
| SQL with parameters | `spark.sql("... WHERE city = :c", args={"c": "Karachi"})` |
| Statistics | `summary()`, `approxQuantile`, `stat.corr`, `stat.crosstab`, `rollup`, `cube`, `pivot` |
| Windows | `Window.partitionBy(..).orderBy(..)` with `lag`, `rank`, `row_number`, `avg().over(w.rowsBetween(-2,0))` |
| Nested data | `from_json`, `explode`, `element_at`, `col("struct.field")`, `aggregate`, `transform` |
| ML pipeline | `Pipeline([VectorAssembler, StandardScaler, KMeans/LinearRegression]).fit(df)` |
| Evaluate | `ClusteringEvaluator` (silhouette) · `RegressionEvaluator` (rmse, mae) |
| Cache / plan | `df.cache()` · `df.explain()` |
| Submit | `spark-submit --master yarn --deploy-mode cluster --num-executors 10 job.py --date 2026-10-24` |

### Key numbers and formulas

| Item | Value |
|---|---|
| HDFS block size | 128 MB (Hadoop 2/3), 64 MB (Hadoop 1) |
| Replication overhead | 3 copies = 200%; erasure coding RS-6-3 = 50% |
| Raw capacity | data × replication × (1 + temp overhead) ÷ max disk fill |
| Heartbeat / dead node | every 3 s / about 10 min without one |
| TF-IDF | (count in doc ÷ words in doc) × log(N ÷ docs containing term) |
| MAE · RMSE · MAPE | mean\|e\| · √mean(e²) · mean\|e ÷ y\| × 100 |

### Which tool when?

| Need | Use |
|---|---|
| Store huge files cheaply and reliably | HDFS (or cloud object storage) |
| Millisecond lookups by key | HBase |
| Import tables from MySQL/Oracle | Sqoop (or Spark JDBC) |
| Stream events and logs | Kafka, Flume |
| Batch SQL reporting and ETL | Hive (on Tez) or Spark SQL |
| Interactive BI queries | Impala, Trino |
| Multi-step scripted ETL on legacy clusters | Pig |
| Fast analytics, ML, streaming | Spark |
| Coordination / scheduling | ZooKeeper / Oozie or Airflow |

## 📝 Assessment

Finish with the [module practice test](practice-test.md): 20 multiple-choice questions and 3 scenario questions with model answers (60 minutes, 50 marks).

---

<sub>Diploma in Data Science · Prepared by **Hamna Zahid Khan**, Junior Lecturer</sub>
