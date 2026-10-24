# 💻 Code · Module 1 · Data Mining with WEKA

> Prepared by **Hamna Zahid Khan**, Junior Lecturer

Every file below is complete and runnable on its own. Each one is used by a session page, which shows the real output it produced.

[⬅ Back to the module](../README.md)

## ⚙️ Setup

Python 3.10+ and these packages:

```bash
pip install mlxtend numpy pandas scikit-learn
```

The `.sh` scripts call WEKA from the command line. Install Java 11+ and download `weka.jar`, then:

```bash
export WEKA_HOME=/path/to/folder/with/weka.jar
cd code
bash session-01_summary.sh
```

Notebooks (`.ipynb`) open directly in Google Colab from the badge at the top of each session page, or locally with `jupyter notebook`.

Run every command from inside this `code/` folder so relative data paths resolve.

## 📂 Files

### [Session 1 · Introduction to Data Mining and WEKA](../session-01-introduction-to-data-mining-and-weka.md)

| File | What it does | How to run |
|---|---|---|
| [`session-01-introduction-to-data-mining-and-weka.ipynb`](session-01-introduction-to-data-mining-and-weka.ipynb) | Executed notebook with every Python example of the session | `jupyter notebook session-01-introduction-to-data-mining-and-weka.ipynb` (or the Colab badge on the session page) |
| [`session-01_csv_to_arff.py`](session-01_csv_to_arff.py) | Example 5 · Converting CSV to ARFF in Python | `python session-01_csv_to_arff.py` |
| [`session-01_first_look.py`](session-01_first_look.py) | Example 4 · The same first look in Python | `python session-01_first_look.py` |
| [`session-01_j48.sh`](session-01_j48.sh) | Example 3 · A first model in one line | `bash session-01_j48.sh` |
| [`session-01_summary.sh`](session-01_summary.sh) | Example 2 · Ask WEKA for a dataset summary | `bash session-01_summary.sh` |
| [`students.arff`](students.arff) | Example 1 · Write your first ARFF file | dataset: open in WEKA Explorer or load from the scripts |

### [Session 2 · Data Warehousing and Data Pre-processing](../session-02-data-warehousing-and-preprocessing.md)

| File | What it does | How to run |
|---|---|---|
| [`employees.arff`](employees.arff) | Example 2 · A dirty dataset | dataset: open in WEKA Explorer or load from the scripts |
| [`session-02-data-warehousing-and-preprocessing.ipynb`](session-02-data-warehousing-and-preprocessing.ipynb) | Executed notebook with every Python example of the session | `jupyter notebook session-02-data-warehousing-and-preprocessing.ipynb` (or the Colab badge on the session page) |
| [`session-02_iqr_outliers.py`](session-02_iqr_outliers.py) | Example 6 · Detecting outliers with the IQR rule | `python session-02_iqr_outliers.py` |
| [`session-02_normalize_discretize.sh`](session-02_normalize_discretize.sh) | Example 4 · Normalise and discretise | `bash session-02_normalize_discretize.sh` |
| [`session-02_preprocess_pandas.py`](session-02_preprocess_pandas.py) | Example 5 · The same pipeline in pandas | `python session-02_preprocess_pandas.py` |
| [`session-02_replace_missing.sh`](session-02_replace_missing.sh) | Example 3 · Replace missing values with a WEKA filter | `bash session-02_replace_missing.sh` |
| [`session-02_star_schema.py`](session-02_star_schema.py) | Example 1 · A mini star schema in SQL | `python session-02_star_schema.py` |

### [Session 3 · Association Rule Mining](../session-03-association-rule-mining.md)

| File | What it does | How to run |
|---|---|---|
| [`session-03-association-rule-mining.ipynb`](session-03-association-rule-mining.ipynb) | Executed notebook with every Python example of the session | `jupyter notebook session-03-association-rule-mining.ipynb` (or the Colab badge on the session page) |
| [`session-03_apriori.sh`](session-03_apriori.sh) | Example 3 · Apriori in WEKA | `bash session-03_apriori.sh` |
| [`session-03_apriori_lift.sh`](session-03_apriori_lift.sh) | Example 4 · Rank by lift instead of confidence | `bash session-03_apriori_lift.sh` |
| [`session-03_measures.py`](session-03_measures.py) | Example 1 · Support, confidence and lift by hand in Python | `python session-03_measures.py` |
| [`session-03_mlxtend_apriori.py`](session-03_mlxtend_apriori.py) | Example 5 · Apriori in Python with mlxtend | `python session-03_mlxtend_apriori.py` |
| [`session-03_tuning.py`](session-03_tuning.py) | Example 6 · How min_support changes the result | `python session-03_tuning.py` |
| [`shop.arff`](shop.arff) | Example 2 · A shop's transactions in ARFF | dataset: open in WEKA Explorer or load from the scripts |

### [Session 4 · Classification and Decision Trees](../session-04-classification-and-decision-trees.md)

| File | What it does | How to run |
|---|---|---|
| [`session-04-classification-and-decision-trees.ipynb`](session-04-classification-and-decision-trees.ipynb) | Executed notebook with every Python example of the session | `jupyter notebook session-04-classification-and-decision-trees.ipynb` (or the Colab badge on the session page) |
| [`session-04_entropy.py`](session-04_entropy.py) | Example 1 · Entropy and information gain from scratch | `python session-04_entropy.py` |
| [`session-04_infogain.sh`](session-04_infogain.sh) | Example 3 · Information gain ranking in WEKA | `bash session-04_infogain.sh` |
| [`session-04_j48_weather.sh`](session-04_j48_weather.sh) | Example 4 · J48 decision tree with 10-fold cross-validation | `bash session-04_j48_weather.sh` |
| [`session-04_metrics.py`](session-04_metrics.py) | Example 6 · Computing the metrics yourself | `python session-04_metrics.py` |
| [`session-04_sklearn_tree.py`](session-04_sklearn_tree.py) | Example 5 · The same tree in scikit-learn | `python session-04_sklearn_tree.py` |
| [`weather.arff`](weather.arff) | Example 2 · The classic weather data | dataset: open in WEKA Explorer or load from the scripts |

### [Session 5 · Regression Techniques](../session-05-regression-techniques.md)

| File | What it does | How to run |
|---|---|---|
| [`houses.arff`](houses.arff) | Example 1 · House-price data | dataset: open in WEKA Explorer or load from the scripts |
| [`session-05-regression-techniques.ipynb`](session-05-regression-techniques.ipynb) | Executed notebook with every Python example of the session | `jupyter notebook session-05-regression-techniques.ipynb` (or the Colab badge on the session page) |
| [`session-05_compare.sh`](session-05_compare.sh) | Example 4 · Comparing methods in WEKA | `bash session-05_compare.sh` |
| [`session-05_linreg.sh`](session-05_linreg.sh) | Example 2 · Linear regression in WEKA | `bash session-05_linreg.sh` |
| [`session-05_m5p.sh`](session-05_m5p.sh) | Example 3 · M5P model tree | `bash session-05_m5p.sh` |
| [`session-05_sklearn_regression.py`](session-05_sklearn_regression.py) | Example 5 · The same comparison in scikit-learn | `python session-05_sklearn_regression.py` |

### [Session 6 · Clustering Techniques](../session-06-clustering-techniques.md)

| File | What it does | How to run |
|---|---|---|
| [`customers.arff`](customers.arff) | Example 1 · Mall-customer data | dataset: open in WEKA Explorer or load from the scripts |
| [`session-06-clustering-techniques.ipynb`](session-06-clustering-techniques.ipynb) | Executed notebook with every Python example of the session | `jupyter notebook session-06-clustering-techniques.ipynb` (or the Colab badge on the session page) |
| [`session-06_classes_to_clusters.sh`](session-06_classes_to_clusters.sh) | Example 7 · Classes-to-clusters evaluation | `bash session-06_classes_to_clusters.sh` |
| [`session-06_em.sh`](session-06_em.sh) | Example 4 · EM chooses k for you | `bash session-06_em.sh` |
| [`session-06_hierarchical.sh`](session-06_hierarchical.sh) | Example 5 · Hierarchical clustering | `bash session-06_hierarchical.sh` |
| [`session-06_k_sweep.sh`](session-06_k_sweep.sh) | Example 3 · Trying different k | `bash session-06_k_sweep.sh` |
| [`session-06_kmeans.sh`](session-06_kmeans.sh) | Example 2 · k-means in WEKA | `bash session-06_kmeans.sh` |
| [`session-06_sklearn_clustering.py`](session-06_sklearn_clustering.py) | Example 6 · Elbow, silhouette and DBSCAN in scikit-learn | `python session-06_sklearn_clustering.py` |

### [Session 7 · Text Mining and Web Mining](../session-07-text-mining-and-web-mining.md)

| File | What it does | How to run |
|---|---|---|
| [`reviews.arff`](reviews.arff) | Example 1 · Product-review data | dataset: open in WEKA Explorer or load from the scripts |
| [`session-07-text-mining-and-web-mining.ipynb`](session-07-text-mining-and-web-mining.ipynb) | Executed notebook with every Python example of the session | `jupyter notebook session-07-text-mining-and-web-mining.ipynb` (or the Colab badge on the session page) |
| [`session-07_pagerank.py`](session-07_pagerank.py) | Example 5 · Web structure mining: PageRank | `python session-07_pagerank.py` |
| [`session-07_sklearn_text.py`](session-07_sklearn_text.py) | Example 4 · TF-IDF, classification and similarity in scikit-learn | `python session-07_sklearn_text.py` |
| [`session-07_text_nb.sh`](session-07_text_nb.sh) | Example 2 · Text classification in WEKA | `bash session-07_text_nb.sh` |
| [`session-07_usage_mining.py`](session-07_usage_mining.py) | Example 6 · Web usage mining: sessions and paths | `python session-07_usage_mining.py` |
| [`session-07_word_vector.sh`](session-07_word_vector.sh) | Example 3 · Looking inside the word vectors | `bash session-07_word_vector.sh` |

### [Session 8 · Designing a Real-World Data Mining Project](../session-08-real-world-data-mining-project.md)

| File | What it does | How to run |
|---|---|---|
| [`churn.arff`](churn.arff) | Step 2 · The data | dataset: open in WEKA Explorer or load from the scripts |
| [`new_customers.arff`](new_customers.arff) | Step 2 · The data | dataset: open in WEKA Explorer or load from the scripts |
| [`session-08-real-world-data-mining-project.ipynb`](session-08-real-world-data-mining-project.ipynb) | Executed notebook with every Python example of the session | `jupyter notebook session-08-real-world-data-mining-project.ipynb` (or the Colab badge on the session page) |
| [`session-08_compare.sh`](session-08_compare.sh) | Step 5 · Compare models with 10-fold cross-validation | `bash session-08_compare.sh` |
| [`session-08_deploy.sh`](session-08_deploy.sh) | Step 6 · Train the final model, save it and score new customers | `bash session-08_deploy.sh` |
| [`session-08_explore.sh`](session-08_explore.sh) | Step 3 · Data understanding in WEKA | `bash session-08_explore.sh` |
| [`session-08_infogain.sh`](session-08_infogain.sh) | Step 4 · Which attributes matter? | `bash session-08_infogain.sh` |
| [`session-08_pipeline.py`](session-08_pipeline.py) | Step 7 · The same project as a scikit-learn pipeline | `python session-08_pipeline.py` |

---

<sub>Prepared by **Hamna Zahid Khan**, Junior Lecturer</sub>
