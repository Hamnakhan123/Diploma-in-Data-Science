# 💻 Code · Module 2 · Foundations of Big Data Systems with Hadoop

> Prepared by **Hamna Zahid Khan**, Junior Lecturer

Every file below is complete and runnable on its own. Each one is used by a session page, which shows the real output it produced.

[⬅ Back to the module](../README.md)

## ⚙️ Setup

Python 3.10+ and these packages:

```bash
pip install pandas plotly pyspark streamlit
```

Notebooks (`.ipynb`) open directly in Google Colab from the badge at the top of each session page, or locally with `jupyter notebook`.

Run every command from inside this `code/` folder so relative data paths resolve.

## 📂 Files

### [Session 1 · Introduction to Big Data and the Hadoop Ecosystem](../session-01-big-data-and-hadoop-ecosystem.md)

| File | What it does | How to run |
|---|---|---|
| [`session-01-big-data-and-hadoop-ecosystem.ipynb`](session-01-big-data-and-hadoop-ecosystem.ipynb) | Executed notebook with every Python example of the session | `jupyter notebook session-01-big-data-and-hadoop-ecosystem.ipynb` (or the Colab badge on the session page) |
| [`session-01_blocks.py`](session-01_blocks.py) | Example 2 · How HDFS splits a file into blocks | `python session-01_blocks.py` |
| [`session-01_scale.py`](session-01_scale.py) | Example 1 · Why scale out? Reading 10 TB | `python session-01_scale.py` |
| [`session-01_wordcount_mr.py`](session-01_wordcount_mr.py) | Example 3 · The MapReduce idea in plain Python | `python session-01_wordcount_mr.py` |

### [Session 2 · Configuring Hadoop and Setting Up a Cluster](../session-02-configuring-hadoop-cluster.md)

| File | What it does | How to run |
|---|---|---|
| [`core-site.xml`](core-site.xml) | Example 1 · The configuration files for a pseudo-distributed cluster | config/data file used by the examples |
| [`hdfs-site.xml`](hdfs-site.xml) | Example 1 · The configuration files for a pseudo-distributed cluster | config/data file used by the examples |
| [`mapred-site.xml`](mapred-site.xml) | Example 1 · The configuration files for a pseudo-distributed cluster | config/data file used by the examples |
| [`session-02-configuring-hadoop-cluster.ipynb`](session-02-configuring-hadoop-cluster.ipynb) | Executed notebook with every Python example of the session | `jupyter notebook session-02-configuring-hadoop-cluster.ipynb` (or the Colab badge on the session page) |
| [`session-02_read_config.py`](session-02_read_config.py) | Example 3 · Reading configuration like Hadoop does | `python session-02_read_config.py` |
| [`session-02_sizing.py`](session-02_sizing.py) | Example 4 · Sizing a cluster | `python session-02_sizing.py` |
| [`yarn-site.xml`](yarn-site.xml) | Example 1 · The configuration files for a pseudo-distributed cluster | config/data file used by the examples |

### [Session 3 · Understanding HDFS and Its Goals](../session-03-hdfs-and-its-goals.md)

| File | What it does | How to run |
|---|---|---|
| [`session-03-hdfs-and-its-goals.ipynb`](session-03-hdfs-and-its-goals.ipynb) | Executed notebook with every Python example of the session | `jupyter notebook session-03-hdfs-and-its-goals.ipynb` (or the Colab badge on the session page) |
| [`session-03_failure_sim.py`](session-03_failure_sim.py) | Example 3 · DataNode failure and re-replication | `python session-03_failure_sim.py` |
| [`session-03_jobtracker_sim.py`](session-03_jobtracker_sim.py) | Example 4 · JobTracker scheduling with data locality | `python session-03_jobtracker_sim.py` |
| [`session-03_namenode_sim.py`](session-03_namenode_sim.py) | Example 2 · A mini NameNode with rack-aware placement | `python session-03_namenode_sim.py` |

### [Session 4 · Exploring YARN and Classic MapReduce](../session-04-yarn-and-classic-mapreduce.md)

| File | What it does | How to run |
|---|---|---|
| [`capacity-scheduler.xml`](capacity-scheduler.xml) | Example 3 · Capacity Scheduler queues | config/data file used by the examples |
| [`session-04-yarn-and-classic-mapreduce.ipynb`](session-04-yarn-and-classic-mapreduce.ipynb) | Executed notebook with every Python example of the session | `jupyter notebook session-04-yarn-and-classic-mapreduce.ipynb` (or the Colab badge on the session page) |
| [`session-04_capacity_sim.py`](session-04_capacity_sim.py) | Example 3 · Capacity Scheduler queues | `python session-04_capacity_sim.py` |
| [`session-04_scheduler_sim.py`](session-04_scheduler_sim.py) | Example 2 · FIFO vs Fair scheduling | `python session-04_scheduler_sim.py` |
| [`session-04_slots_vs_containers.py`](session-04_slots_vs_containers.py) | Example 1 · Slots vs containers | `python session-04_slots_vs_containers.py` |

### [Session 5 · Demonstrating MapReduce Algorithms](../session-05-mapreduce-algorithms.md)

| File | What it does | How to run |
|---|---|---|
| [`session-05-mapreduce-algorithms.ipynb`](session-05-mapreduce-algorithms.ipynb) | Executed notebook with every Python example of the session | `jupyter notebook session-05-mapreduce-algorithms.ipynb` (or the Colab badge on the session page) |
| [`session-05_mr_engine.py`](session-05_mr_engine.py) | Example 1 · A reusable MapReduce simulator with a combiner | `python session-05_mr_engine.py` |
| [`session-05_mr_grep.py`](session-05_mr_grep.py) | Example 2 · Search: distributed grep (map-only) | `python session-05_mr_grep.py` |
| [`session-05_mr_index.py`](session-05_mr_index.py) | Example 4 · Building an inverted index | `python session-05_mr_index.py` |
| [`session-05_mr_sort.py`](session-05_mr_sort.py) | Example 3 · Total-order sort with a range partitioner | `python session-05_mr_sort.py` |
| [`session-05_mr_tfidf.py`](session-05_mr_tfidf.py) | Example 5 · TF-IDF as three chained MapReduce jobs | `python session-05_mr_tfidf.py` |
| [`session-05_spark_tfidf.py`](session-05_spark_tfidf.py) | Example 6 · The same TF-IDF in Spark (real PySpark) | `python session-05_spark_tfidf.py` |

### [Session 6 · Understanding Sqoop and Joins in MapReduce](../session-06-sqoop-and-mapreduce-joins.md)

| File | What it does | How to run |
|---|---|---|
| [`session-06-sqoop-and-mapreduce-joins.ipynb`](session-06-sqoop-and-mapreduce-joins.ipynb) | Executed notebook with every Python example of the session | `jupyter notebook session-06-sqoop-and-mapreduce-joins.ipynb` (or the Colab badge on the session page) |
| [`session-06_map_join.py`](session-06_map_join.py) | Example 4 · Map-side (replicated) join | `python session-06_map_join.py` |
| [`session-06_reduce_join.py`](session-06_reduce_join.py) | Example 3 · Reduce-side join | `python session-06_reduce_join.py` |
| [`session-06_spark_joins.py`](session-06_spark_joins.py) | Example 5 · Joins in Spark: broadcast vs sort-merge (real PySpark) | `python session-06_spark_joins.py` |
| [`session-06_sqoop_incremental.py`](session-06_sqoop_incremental.py) | Example 2 · Incremental append import | `python session-06_sqoop_incremental.py` |
| [`session-06_sqoop_sim.py`](session-06_sqoop_sim.py) | Example 1 · How Sqoop splits an import (simulated with SQLite) | `python session-06_sqoop_sim.py` |

### [Session 7 · Exploring Hadoop Architecture and Components](../session-07-hadoop-architecture-and-components.md)

| File | What it does | How to run |
|---|---|---|
| [`session-07-hadoop-architecture-and-components.ipynb`](session-07-hadoop-architecture-and-components.ipynb) | Executed notebook with every Python example of the session | `jupyter notebook session-07-hadoop-architecture-and-components.ipynb` (or the Colab badge on the session page) |
| [`session-07_ec_vs_replication.py`](session-07_ec_vs_replication.py) | Example 1 · Replication vs erasure coding | `python session-07_ec_vs_replication.py` |
| [`session-07_ha_failover.py`](session-07_ha_failover.py) | Example 3 · NameNode failover with a journal quorum | `python session-07_ha_failover.py` |
| [`session-07_hbase_rowkeys.py`](session-07_hbase_rowkeys.py) | Example 4 · HBase-style row keys | `python session-07_hbase_rowkeys.py` |
| [`session-07_xor_parity.py`](session-07_xor_parity.py) | Example 2 · Parity in action (the idea behind erasure coding) | `python session-07_xor_parity.py` |

### [Session 8 · Working with Hive and Impala](../session-08-hive-and-impala.md)

| File | What it does | How to run |
|---|---|---|
| [`session-08-hive-and-impala.ipynb`](session-08-hive-and-impala.ipynb) | Executed notebook with every Python example of the session | `jupyter notebook session-08-hive-and-impala.ipynb` (or the Colab badge on the session page) |
| [`session-08_bucketing.py`](session-08_bucketing.py) | Example 3 · Bucketing and table sampling | `python session-08_bucketing.py` |
| [`session-08_hive_advanced.py`](session-08_hive_advanced.py) | Example 4 · Complex types, explode and window functions | `python session-08_hive_advanced.py` |
| [`session-08_managed_external.py`](session-08_managed_external.py) | Example 1 · Managed vs external tables | `python session-08_managed_external.py` |
| [`session-08_partitioning.py`](session-08_partitioning.py) | Example 2 · Partitioning and partition pruning | `python session-08_partitioning.py` |

### [Session 9 · Introduction to Apache Pig](../session-09-introduction-to-apache-pig.md)

| File | What it does | How to run |
|---|---|---|
| [`session-09-introduction-to-apache-pig.ipynb`](session-09-introduction-to-apache-pig.ipynb) | Executed notebook with every Python example of the session | `jupyter notebook session-09-introduction-to-apache-pig.ipynb` (or the Colab badge on the session page) |
| [`session-09_line_counts.py`](session-09_line_counts.py) | Example 3 · Same question, four languages | `python session-09_line_counts.py` |
| [`session-09_pig_in_spark.py`](session-09_pig_in_spark.py) | Example 2 · The same data flow in PySpark | `python session-09_pig_in_spark.py` |
| [`session-09_pig_trace.py`](session-09_pig_trace.py) | Example 1 · Tracing the script step by step (Pig semantics in Python) | `python session-09_pig_trace.py` |

### [Session 10 · Pig Latin Data Model and Operators](../session-10-pig-latin-data-model-and-operators.md)

| File | What it does | How to run |
|---|---|---|
| [`session-10-pig-latin-data-model-and-operators.ipynb`](session-10-pig-latin-data-model-and-operators.ipynb) | Executed notebook with every Python example of the session | `jupyter notebook session-10-pig-latin-data-model-and-operators.ipynb` (or the Colab badge on the session page) |
| [`session-10_nested_foreach.py`](session-10_nested_foreach.py) | Example 4 · Nested FOREACH: top 2 per group | `python session-10_nested_foreach.py` |
| [`session-10_pig_data_model.py`](session-10_pig_data_model.py) | Example 1 · Tuples, bags, GROUP, COGROUP and FLATTEN in Python | `python session-10_pig_data_model.py` |
| [`session-10_pig_logs_spark.py`](session-10_pig_logs_spark.py) | Example 3 · The same analysis, run for real in PySpark | `python session-10_pig_logs_spark.py` |
| [`session-10_pig_nulls.py`](session-10_pig_nulls.py) | Example 5 · Nulls, casting and DISTINCT | `python session-10_pig_nulls.py` |

### [Session 11 · Introduction to Apache Spark](../session-11-introduction-to-apache-spark.md)

| File | What it does | How to run |
|---|---|---|
| [`session-11-introduction-to-apache-spark.ipynb`](session-11-introduction-to-apache-spark.ipynb) | Executed notebook with every Python example of the session | `jupyter notebook session-11-introduction-to-apache-spark.ipynb` (or the Colab badge on the session page) |
| [`session-11_cache_stages.py`](session-11_cache_stages.py) | Example 4 · Jobs, stages and caching | `python session-11_cache_stages.py` |
| [`session-11_dataframes.py`](session-11_dataframes.py) | Example 3 · DataFrames, SQL and the Catalyst plan | `python session-11_dataframes.py` |
| [`session-11_lazy.py`](session-11_lazy.py) | Example 2 · Lazy evaluation | `python session-11_lazy.py` |
| [`session-11_rdd_wordcount.py`](session-11_rdd_wordcount.py) | Example 1 · Word count with RDDs | `python session-11_rdd_wordcount.py` |

### [Session 12 · Spark Statistics for Interactive Dashboards](../session-12-spark-statistics-and-dashboards.md)

| File | What it does | How to run |
|---|---|---|
| [`dashboard_app.py`](dashboard_app.py) | Example 5 · An interactive dashboard app (Streamlit) | `python dashboard_app.py` |
| [`session-12-spark-statistics-and-dashboards.ipynb`](session-12-spark-statistics-and-dashboards.ipynb) | Executed notebook with every Python example of the session | `jupyter notebook session-12-spark-statistics-and-dashboards.ipynb` (or the Colab badge on the session page) |
| [`session-12_spark_agg_table.py`](session-12_spark_agg_table.py) | Example 4 · Writing the dashboard's aggregate table | `python session-12_spark_agg_table.py` |
| [`session-12_spark_kpis.py`](session-12_spark_kpis.py) | Example 2 · KPI tables with groupBy, rollup and pivot | `python session-12_spark_kpis.py` |
| [`session-12_spark_stats.py`](session-12_spark_stats.py) | Example 1 · Descriptive statistics at scale | `python session-12_spark_stats.py` |
| [`session-12_spark_windows.py`](session-12_spark_windows.py) | Example 3 · Trends with window functions | `python session-12_spark_windows.py` |

### [Session 13 · Data Analysis with Spark: Clustering and Forecasting](../session-13-spark-clustering-and-forecasting.md)

| File | What it does | How to run |
|---|---|---|
| [`session-13-spark-clustering-and-forecasting.ipynb`](session-13-spark-clustering-and-forecasting.ipynb) | Executed notebook with every Python example of the session | `jupyter notebook session-13-spark-clustering-and-forecasting.ipynb` (or the Colab badge on the session page) |
| [`session-13_customer_features.py`](session-13_customer_features.py) | Example 1 · Data preparation: customer features | `python session-13_customer_features.py` |
| [`session-13_forecast.py`](session-13_forecast.py) | Example 3 · Forecasting daily sales with lag features | `python session-13_forecast.py` |
| [`session-13_kmeans.py`](session-13_kmeans.py) | Example 2 · K-means segmentation with a Pipeline and silhouette | `python session-13_kmeans.py` |

### [Session 14 · Special Field Types, Generated Fields and Parameters in Spark](../session-14-spark-field-types-and-parameters.md)

| File | What it does | How to run |
|---|---|---|
| [`session-14-spark-field-types-and-parameters.ipynb`](session-14-spark-field-types-and-parameters.ipynb) | Executed notebook with every Python example of the session | `jupyter notebook session-14-spark-field-types-and-parameters.ipynb` (or the Colab badge on the session page) |
| [`session-14_generated_fields.py`](session-14_generated_fields.py) | Example 3 · Generated fields | `python session-14_generated_fields.py` |
| [`session-14_nested_json.py`](session-14_nested_json.py) | Example 2 · Nested JSON: structs, arrays and maps | `python session-14_nested_json.py` |
| [`session-14_parameters.py`](session-14_parameters.py) | Example 4 · Parameterised SQL, broadcast variables and accumulators | `python session-14_parameters.py` |
| [`session-14_special_types.py`](session-14_special_types.py) | Example 1 · Dates, timestamps, time zones and decimals | `python session-14_special_types.py` |

### [Session 15 · Final Project and Next Steps](../session-15-final-project-and-next-steps.md)

| File | What it does | How to run |
|---|---|---|
| [`session-15-final-project-and-next-steps.ipynb`](session-15-final-project-and-next-steps.ipynb) | Executed notebook with every Python example of the session | `jupyter notebook session-15-final-project-and-next-steps.ipynb` (or the Colab badge on the session page) |
| [`session-15_pipeline.py`](session-15_pipeline.py) | Example 1 · Reference pipeline: bronze → silver → gold | `python session-15_pipeline.py` |
| [`session-15_project_model.py`](session-15_project_model.py) | Example 2 · Model and data-quality checks on the silver table | `python session-15_project_model.py` |

---

<sub>Prepared by **Hamna Zahid Khan**, Junior Lecturer</sub>
