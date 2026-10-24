# 📝 Practice Test · Foundations of Big Data Systems with Hadoop

> 📍 [Diploma in Data Science](../../README.md) › [Term 2](../README.md) › [Foundations of Big Data Systems with Hadoop](README.md) › **Practice test**

Attempt this after finishing all 15 sessions. Suggested time: **60 minutes** (20 MCQs × 1 mark, 3 scenarios × 10 marks = 50 marks). Try every question before opening an answer.

[⬅ Back to Session 15](session-15-final-project-and-next-steps.md) · [🏠 Module home](README.md)

---

## Part A · Multiple-choice questions (20 marks)

**Q1.** Which of the 5 Vs refers to how trustworthy and clean the data is?

- **A)** Volume
- **B)** Velocity
- **C)** Veracity
- **D)** Value

<details><summary>Show answer</summary>

**C** — Veracity is about quality and reliability.

</details>

**Q2.** A 500 MB file stored in HDFS with 128 MB blocks and replication 3 uses how many block replicas?

- **A)** 4
- **B)** 12
- **C)** 15
- **D)** 3

<details><summary>Show answer</summary>

**B** — ceil(500 ÷ 128) = 4 blocks × 3 replicas = 12.

</details>

**Q3.** Which property in core-site.xml tells clients where the file system lives?

- **A)** dfs.replication
- **B)** fs.defaultFS
- **C)** yarn.resourcemanager.hostname
- **D)** mapreduce.framework.name

<details><summary>Show answer</summary>

**B** — For example hdfs://master:9000.

</details>

**Q4.** The Secondary NameNode…

- **A)** takes over automatically when the NameNode fails
- **B)** merges fsimage and edits into a checkpoint
- **C)** stores a fourth replica of each block
- **D)** schedules map tasks

<details><summary>Show answer</summary>

**B** — It is a checkpointing helper, not a hot standby.

</details>

**Q5.** With the default rack-aware policy and replication 3, the replicas are spread over…

- **A)** one rack
- **B)** two racks
- **C)** three racks
- **D)** every rack

<details><summary>Show answer</summary>

**B** — One on the writer's rack, two on one remote rack.

</details>

**Q6.** In YARN, which component negotiates containers for a single application?

- **A)** ResourceManager
- **B)** NodeManager
- **C)** ApplicationMaster
- **D)** JobTracker

<details><summary>Show answer</summary>

**C** — Each application has its own ApplicationMaster.

</details>

**Q7.** Which scheduler guarantees each department queue a percentage of the cluster?

- **A)** FIFO
- **B)** Capacity
- **C)** Fair
- **D)** Random

<details><summary>Show answer</summary>

**B** — The Capacity Scheduler uses queues with guaranteed shares.

</details>

**Q8.** A combiner can safely be used to compute…

- **A)** an average directly
- **B)** a sum
- **C)** a median
- **D)** a distinct list with ordering

<details><summary>Show answer</summary>

**B** — Combiners need associative and commutative operations.

</details>

**Q9.** Total-order sorting across many reducers needs…

- **A)** a hash partitioner
- **B)** a range partitioner with sampled split points
- **C)** zero reducers
- **D)** a combiner

<details><summary>Show answer</summary>

**B** — Like TeraSort's TotalOrderPartitioner.

</details>

**Q10.** A Sqoop import with `-m 4` runs…

- **A)** 4 reducers
- **B)** 4 parallel map tasks, each importing a range of the split column
- **C)** 4 Hive queries
- **D)** 4 separate databases

<details><summary>Show answer</summary>

**B** — Sqoop imports are map-only jobs.

</details>

**Q11.** A map-side (replicated) join is possible when…

- **A)** both tables are huge
- **B)** one table fits in memory on every mapper
- **C)** the tables have no common key
- **D)** a combiner is defined

<details><summary>Show answer</summary>

**B** — The small table is cached on every node; no shuffle.

</details>

**Q12.** HDFS NameNode HA shares edits between active and standby through…

- **A)** DataNodes
- **B)** a quorum of JournalNodes
- **C)** the Secondary NameNode
- **D)** Oozie

<details><summary>Show answer</summary>

**B** — The standby tails the edit log from the JournalNodes.

</details>

**Q13.** Erasure coding RS-6-3 tolerates how many lost cells, with what overhead?

- **A)** 2, 200%
- **B)** 3, 50%
- **C)** 3, 200%
- **D)** 6, 33%

<details><summary>Show answer</summary>

**B** — Any 6 of 9 cells rebuild the data; 3 ÷ 6 = 50% overhead.

</details>

**Q14.** Dropping an EXTERNAL Hive table…

- **A)** deletes the data files
- **B)** removes only the metadata
- **C)** is not allowed
- **D)** converts it to managed

<details><summary>Show answer</summary>

**B** — External data is owned by you, not Hive.

</details>

**Q15.** Partition pruning happens when…

- **A)** the WHERE clause filters on a partition column
- **B)** a table is bucketed
- **C)** data is in CSV
- **D)** a broadcast join is used

<details><summary>Show answer</summary>

**A** — Only matching partition folders are read.

</details>

**Q16.** In Pig, `GROUP students BY city` produces tuples of…

- **A)** (city, count)
- **B)** (city, bag of student tuples)
- **C)** one tuple per student
- **D)** a map

<details><summary>Show answer</summary>

**B** — FOREACH then aggregates each bag.

</details>

**Q17.** Which Pig command shows sample data flowing through each step of a script?

- **A)** DESCRIBE
- **B)** EXPLAIN
- **C)** ILLUSTRATE
- **D)** DUMP

<details><summary>Show answer</summary>

**C** — ILLUSTRATE runs the flow on a small sample.

</details>

**Q18.** In Spark, a new stage begins at…

- **A)** every transformation
- **B)** every shuffle (wide dependency)
- **C)** every filter
- **D)** every cache

<details><summary>Show answer</summary>

**B** — Narrow transformations are pipelined within a stage.

</details>

**Q19.** Before K-means in Spark MLlib you should…

- **A)** one-hot encode the label
- **B)** assemble features into a vector and scale them
- **C)** sort by the cluster column
- **D)** call collect()

<details><summary>Show answer</summary>

**B** — K-means needs a feature vector and is distance-based.

</details>

**Q20.** Why use `spark.sql(query, args={"city": value})` instead of an f-string?

- **A)** It is faster to type
- **B)** Values are bound safely, preventing SQL injection
- **C)** It caches the result
- **D)** It runs on YARN only

<details><summary>Show answer</summary>

**B** — Bound parameters are never parsed as SQL.

</details>


## Part B · Scenario questions (30 marks, 10 each)

### 🧩 Scenario 1 · Telecom call records platform

A mobile operator in Pakistan produces 2 TB of call detail records (CDRs) per day and must keep 18 months online. Records are also needed by the billing team's Oracle database, and analysts want daily SQL reports and an interactive dashboard of dropped calls by city and hour.

1. Estimate the raw HDFS capacity needed with replication 3, 25% temporary overhead and disks kept below 80% full. Show your working. (3)
2. Propose how to reduce that storage without losing fault tolerance. (2)
3. Design the Hive table: partitioning, bucketing and file format. Justify each choice. (3)
4. Which engine would you put behind the dashboard, and why? (2)

<details><summary>Show model solution</summary>

1. Data = 2 TB × ~548 days (18 months) ≈ **1,096 TB**. Raw = 1,096 × 3 × 1.25 ÷ 0.8 ≈ **5,138 TB (about 5.1 PB)**.
2. Use **erasure coding** (for example RS-6-3, 50% overhead instead of 200%) for older, cold partitions while keeping recent hot data replicated; store data in a **compressed columnar format** (ORC or Parquet with Snappy/ZSTD), which typically shrinks CDRs several times. Together these cut raw storage by well over half while still tolerating multiple disk failures.
3. `PARTITIONED BY (call_date DATE)`: almost every query filters by date, giving partition pruning, and old partitions can be dropped or moved to EC easily. Possibly a second level by region if queries always filter on it, but **not** by subscriber (too many partitions and tiny files). `CLUSTERED BY (msisdn) INTO 64 BUCKETS`: subscriber ID is high-cardinality and used in joins with billing/customer tables, enabling bucket map joins and sampling. **ORC or Parquet**: columnar, compressed, predicate push-down, and ORC supports ACID updates if corrections are needed.
4. **Impala** (or Trino/Spark SQL on a pre-aggregated table): long-running MPP daemons give sub-second to seconds latency for interactive filters, and it shares the Hive metastore. Best practice is to point the dashboard at a **daily aggregate table** (dropped calls by city, hour, date) produced by a scheduled Hive/Spark job rather than raw CDRs.

</details>


### 🧩 Scenario 2 · Slow nightly MapReduce pipeline

A retail company runs a nightly chain: Sqoop imports 40 million orders from MySQL, a MapReduce job joins orders with a 5,000-row product table and another joins with 20 million customers, then a third job computes revenue per city. The whole chain takes 6 hours and often misses the 6 am deadline. One reducer always runs far longer than the others.

1. Identify three likely causes of the slowness. (3)
2. How would you fix the product-table join? (2)
3. What is the slow reducer a symptom of, and how can it be reduced? (2)
4. Outline how you would re-implement the pipeline in Spark and why it should be faster. (3)

<details><summary>Show model solution</summary>

1. (a) The **product join is done reduce-side**, shuffling 40 million orders just to attach a tiny table. (b) **Chained MapReduce jobs** write intermediate results to HDFS and re-read them each step. (c) **Data skew**: one join key (for example a "walk-in" customer ID or a big city) sends far more records to one reducer. Other valid answers: Sqoop using too few mappers or a skewed `--split-by` column; full rather than incremental imports; no combiner in the aggregation.
2. Use a **map-side (replicated) join**: put the 5,000-row product table in the Distributed Cache and join in the mapper, with no shuffle. In Spark: `orders.join(F.broadcast(products), "product_id")`.
3. **Key skew**: the default hash partitioner sends every record of a hot key to the same reducer. Fixes: **salt** the hot key (append a random suffix, aggregate twice), handle the hot key separately, use a **combiner** to pre-aggregate, or in Spark enable **Adaptive Query Execution skew-join handling** (`spark.sql.adaptive.skewJoin.enabled`).
4. Import incrementally (Sqoop `--incremental append` or Spark JDBC with partitioned reads by `id`), then one Spark job: read orders, **broadcast-join** products, sort-merge join customers (or bucket both on `customer_id`), `groupBy("city").agg(sum)`, and write a partitioned Parquet gold table. It is faster because intermediate data stays **in memory** across the steps, Catalyst optimises the whole plan (filter and column pruning), AQE handles skew and adapts partitions, and there is one job instead of three separate ones writing to HDFS.

</details>


### 🧩 Scenario 3 · Customer segmentation and demand forecasting with Spark

An online grocery service in Karachi and Lahore has 3 years of order data (150 million rows: customer, timestamp, area, basket value, items as a JSON array). Management wants (a) customer segments for targeted offers and (b) a daily order forecast per city for staffing riders.

1. Describe the data preparation in Spark, including how you would handle the JSON items column and time zones. (3)
2. Explain how you would build and choose the segmentation model. (3)
3. Explain how you would build and evaluate the forecast so the results are trustworthy. (4)

<details><summary>Show model solution</summary>

1. Read with an **explicit schema**; set `spark.sql.session.timeZone` to `Asia/Karachi` and convert timestamps with `to_timestamp`, deriving `order_date`, hour and day of week. Parse items with **`from_json`** into `ARRAY<STRUCT<sku, qty, price>>`; use `size()` and `aggregate()` for item counts and basket checks, or `explode` for SKU-level analysis. Drop duplicates and invalid values (negative baskets, null customers), store **silver data partitioned by date** in Parquet, and build per-customer features (recency, frequency, monetary value, average basket, share of fresh items) with `groupBy`.
2. `VectorAssembler` → **`StandardScaler`** (features have very different ranges) → **K-means** inside an MLlib `Pipeline`. Try k = 2…8 and pick k using the **silhouette score** (with the elbow of training cost as a second view) plus business sense (segments must be actionable and not too small). Profile each cluster (average recency, frequency, value) and name it, for example "loyal high-value", "lapsing", "bargain weekend shoppers". Validate stability by re-running with different seeds or on a different month.
3. Aggregate to **daily orders per city**. Create features with **window functions that only look backwards**: `lag_1`, `lag_7`, 7-day rolling mean, day of week, month, holiday and Ramadan/Eid flags, trend. **Split by time** (for example train on the first 33 months and test on the last 3), never randomly. Fit models (linear regression with trend, gradient-boosted trees) and compare with a **naive baseline** (same day last week) using **MAE and RMSE** (and MAPE, since order counts are never near zero). Check errors around holidays separately, retrain regularly, and remember tree models cannot extrapolate a growth trend. For many areas, use `groupBy("area").applyInPandas` to fit one small model per series in parallel.

</details>

---

<sub>Diploma in Data Science · Prepared by **Hamna Zahid Khan**, Junior Lecturer</sub>
