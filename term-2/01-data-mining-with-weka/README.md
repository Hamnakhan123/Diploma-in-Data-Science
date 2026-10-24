# Module 1 · Data Mining with WEKA

> 📍 [Diploma in Data Science](../../README.md) › [Term 2](../README.md) › **Data Mining with WEKA**

⏱ **24 hours** · 8 sessions · Prepared by **Hamna Zahid Khan**, Junior Lecturer

[⬅ Term 2 overview](../README.md) · [▶ Start Session 1](session-01-introduction-to-data-mining-and-weka.md) · [Foundations of Big Data Systems with Hadoop ➡](../02-big-data-systems-with-hadoop/README.md)

---

## 🎯 About this module

This module teaches the full data mining workflow with **WEKA**, the free Java workbench from the University of Waikato, and shows the same ideas in **Python (pandas and scikit-learn)**. You will prepare data, discover association rules, build classifiers and regression models, find clusters, mine text and web data, and finish by running a complete project with the CRISP-DM process.

**By the end you will be able to:**

- explain the KDD and CRISP-DM processes and where data mining fits;
- clean, transform and reduce data, and design a simple star schema;
- mine association rules and judge them with support, confidence and lift;
- build, evaluate and compare classification and regression models;
- cluster data and choose the number of clusters sensibly;
- turn text into features and analyse links and logs on the web;
- plan, run and report a real-world data mining project.

## 📚 Sessions

| # | Session |
|---|---|
| 1 | [Introduction to Data Mining and WEKA](session-01-introduction-to-data-mining-and-weka.md) |
| 2 | [Data Warehousing and Data Pre-processing](session-02-data-warehousing-and-preprocessing.md) |
| 3 | [Association Rule Mining](session-03-association-rule-mining.md) |
| 4 | [Classification and Decision Trees](session-04-classification-and-decision-trees.md) |
| 5 | [Regression Techniques](session-05-regression-techniques.md) |
| 6 | [Clustering Techniques](session-06-clustering-techniques.md) |
| 7 | [Text Mining and Web Mining](session-07-text-mining-and-web-mining.md) |
| 8 | [Designing a Real-World Data Mining Project](session-08-real-world-data-mining-project.md) |
| 📝 | [Module practice test](practice-test.md) |

## 🛠️ Setup

1. Install **Java 11 or later** and **WEKA 3.8** from [the WEKA website](https://ml.cms.waikato.ac.nz/weka). The installer includes the Explorer GUI used in every session.
2. For the command-line scripts in `code/*.sh`, set `WEKA_HOME` to the folder that contains `weka.jar`, then run `bash session-XX_name.sh` from inside `code/`.
3. For the Python examples: `pip install pandas numpy scikit-learn mlxtend matplotlib`, or open the session notebooks in Google Colab with the badge at the top of each session.

All datasets are small ARFF files in [`code/`](code/) so that every result in the notes can be reproduced exactly.

## 📋 Cheat sheet

### WEKA command line

| Task | Command |
|---|---|
| Dataset summary | `java -cp weka.jar weka.core.Instances data.arff` |
| CSV to ARFF | `java -cp weka.jar weka.core.converters.CSVLoader data.csv > data.arff` |
| Classifier with 10-fold CV | `java -cp weka.jar weka.classifiers.trees.J48 -t data.arff` |
| Train/test files | `... -t train.arff -T test.arff` |
| Save / load a model | `... -t data.arff -d model.model` · `... -l model.model -T new.arff` |
| Show predictions | `... -classifications weka.classifiers.evaluation.output.prediction.PlainText` |
| Apply a filter | `java -cp weka.jar weka.filters.unsupervised.attribute.Normalize -i in.arff -o out.arff` |
| Association rules | `java -cp weka.jar weka.associations.Apriori -t data.arff -N 10 -C 0.8` |
| Clustering | `java -cp weka.jar weka.clusterers.SimpleKMeans -t data.arff -N 3` |
| Attribute ranking | `java -cp weka.jar weka.attributeSelection.InfoGainAttributeEval -i data.arff -s weka.attributeSelection.Ranker` |

### Which algorithm?

| Goal | WEKA | scikit-learn |
|---|---|---|
| Baseline | `rules.ZeroR` | `DummyClassifier` |
| Explainable classifier | `trees.J48`, `rules.JRip` | `DecisionTreeClassifier` |
| Fast probabilistic classifier | `bayes.NaiveBayes`, `NaiveBayesMultinomial` (text) | `GaussianNB`, `MultinomialNB` |
| Strong general classifier | `trees.RandomForest`, `functions.SMO` | `RandomForestClassifier`, `SVC` |
| Linear regression | `functions.LinearRegression` | `LinearRegression` |
| Non-linear regression | `trees.M5P`, `functions.SMOreg` | `DecisionTreeRegressor`, `SVR` |
| Association rules | `associations.Apriori`, `FPGrowth` | `mlxtend.frequent_patterns.apriori` |
| Clustering | `clusterers.SimpleKMeans`, `EM`, `HierarchicalClusterer` | `KMeans`, `GaussianMixture`, `AgglomerativeClustering`, `DBSCAN` |
| Text features | `StringToWordVector` | `TfidfVectorizer` |

### Key formulas

| Measure | Formula |
|---|---|
| Support (A → B) | count(A and B) ÷ N |
| Confidence (A → B) | support(A and B) ÷ support(A) |
| Lift (A → B) | confidence(A → B) ÷ support(B); above 1 = positive association |
| Entropy | −Σ pᵢ log₂ pᵢ |
| Information gain | entropy(parent) − weighted entropy(children) |
| Accuracy | (TP + TN) ÷ total |
| Precision · Recall | TP ÷ (TP + FP) · TP ÷ (TP + FN) |
| F1 | 2 × precision × recall ÷ (precision + recall) |
| MAE · RMSE | mean \|error\| · √mean(error²) |
| Min-max normalisation | (x − min) ÷ (max − min) |
| z-score | (x − mean) ÷ standard deviation |
| TF-IDF | tf(t, d) × log(N ÷ df(t)) |
| PageRank | (1 − d) ÷ N + d × Σ PR(q) ÷ out(q) |

### Evaluation rules of thumb

- Always compare against **ZeroR** (majority class) or the mean predictor.
- Use **10-fold cross-validation** on small data; keep a final hold-out set on large data.
- Pick the metric that matches the **business cost**: recall for missed churners or fraud, precision for costly interventions, MAE for numeric targets in real units.
- Put pre-processing **inside** cross-validation (`FilteredClassifier` in WEKA, `Pipeline` in scikit-learn) to avoid leakage.

## 📝 Assessment

Finish with the [module practice test](practice-test.md): 20 multiple-choice questions and 3 scenario questions with model answers (60 minutes, 50 marks).

---

<sub>Diploma in Data Science · Prepared by **Hamna Zahid Khan**, Junior Lecturer</sub>
