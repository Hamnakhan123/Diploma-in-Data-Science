# 💻 Code · Module 3 · Advanced Data Science with Python

> Prepared by **Hamna Zahid Khan**, Junior Lecturer

Every file below is complete and runnable on its own. Each one is used by a session page, which shows the real output it produced.

[⬅ Back to the module](../README.md)

## ⚙️ Setup

Python 3.10+ and these packages:

```bash
pip install joblib lime matplotlib mlxtend numpy onnxruntime pandas scikit-learn scipy seaborn shap skl2onnx statsmodels torch
```

Notebooks (`.ipynb`) open directly in Google Colab from the badge at the top of each session page, or locally with `jupyter notebook`.

Run every command from inside this `code/` folder so relative data paths resolve.

## 📂 Files

### [Session 1 · Fundamentals of Advanced Data Science](../session-01-fundamentals-of-advanced-data-science.md)

| File | What it does | How to run |
|---|---|---|
| [`session-01-fundamentals-of-advanced-data-science.ipynb`](session-01-fundamentals-of-advanced-data-science.ipynb) | Executed notebook with every Python example of the session | `jupyter notebook session-01-fundamentals-of-advanced-data-science.ipynb` (or the Colab badge on the session page) |
| [`session-01_eda_dashboard.png`](session-01_eda_dashboard.png) | Example 3 · Intermediate: a multi-panel Matplotlib + Seaborn figure | image produced by the matching script |
| [`session-01_eda_figure.py`](session-01_eda_figure.py) | Example 3 · Intermediate: a multi-panel Matplotlib + Seaborn figure | `python session-01_eda_figure.py` |
| [`session-01_mini_project.py`](session-01_mini_project.py) | 🧩 Mini project: data science readiness report | `python session-01_mini_project.py` |
| [`session-01_mini_project_profile.png`](session-01_mini_project_profile.png) | 🧩 Mini project: data science readiness report | image produced by the matching script |
| [`session-01_pandas_power.py`](session-01_pandas_power.py) | Example 2 · Intermediate: pandas method chaining, groupby, merge and pivot | `python session-01_pandas_power.py` |
| [`session-01_shop_insights.png`](session-01_shop_insights.png) | 🏙️ Real-life data problem: a corner shop's stock decisions | image produced by the matching script |
| [`session-01_shop_problem.py`](session-01_shop_problem.py) | 🏙️ Real-life data problem: a corner shop's stock decisions | `python session-01_shop_problem.py` |
| [`session-01_sklearn_api.py`](session-01_sklearn_api.py) | Example 4 · Advanced: the scikit-learn API and a Pipeline | `python session-01_sklearn_api.py` |
| [`session-01_three_types.py`](session-01_three_types.py) | Example 5 · Advanced: one data set, three kinds of learning | `python session-01_three_types.py` |
| [`session-01_vectorisation.py`](session-01_vectorisation.py) | Example 1 · Basic: vectorisation vs loops, and broadcasting | `python session-01_vectorisation.py` |

### [Session 2 · Supervised Learning](../session-02-supervised-learning.md)

| File | What it does | How to run |
|---|---|---|
| [`session-02-supervised-learning.ipynb`](session-02-supervised-learning.ipynb) | Executed notebook with every Python example of the session | `jupyter notebook session-02-supervised-learning.ipynb` (or the Colab badge on the session page) |
| [`session-02_gd_scratch.py`](session-02_gd_scratch.py) | Example 1 · Basic: linear regression by gradient descent from scratch | `python session-02_gd_scratch.py` |
| [`session-02_gradient_descent.png`](session-02_gradient_descent.png) | Example 1 · Basic: linear regression by gradient descent from scratch | image produced by the matching script |
| [`session-02_knn.py`](session-02_knn.py) | Example 3 · Intermediate: KNN from scratch, then choosing k | `python session-02_knn.py` |
| [`session-02_learning_rate.py`](session-02_learning_rate.py) | Example 2 · Intermediate: the learning rate and scikit-learn | `python session-02_learning_rate.py` |
| [`session-02_loan_problem.py`](session-02_loan_problem.py) | 🏙️ Real-life data problem: predicting loan default | `python session-02_loan_problem.py` |
| [`session-02_logistic.py`](session-02_logistic.py) | Example 5 · Advanced: logistic regression, the sigmoid and thresholds | `python session-02_logistic.py` |
| [`session-02_logistic_sigmoid.png`](session-02_logistic_sigmoid.png) | Example 5 · Advanced: logistic regression, the sigmoid and thresholds | image produced by the matching script |
| [`session-02_mini_project.py`](session-02_mini_project.py) | 🧩 Mini project: student pass predictor | `python session-02_mini_project.py` |
| [`session-02_naive_bayes.py`](session-02_naive_bayes.py) | Example 4 · Intermediate: Naive Bayes spam filter | `python session-02_naive_bayes.py` |

### [Session 3 · Advanced Supervised Learning: Regression Techniques](../session-03-advanced-regression-techniques.md)

| File | What it does | How to run |
|---|---|---|
| [`session-03-advanced-regression-techniques.ipynb`](session-03-advanced-regression-techniques.ipynb) | Executed notebook with every Python example of the session | `jupyter notebook session-03-advanced-regression-techniques.ipynb` (or the Colab badge on the session page) |
| [`session-03_car_predictions.png`](session-03_car_predictions.png) | 🏙️ Real-life data problem: pricing used cars | image produced by the matching script |
| [`session-03_car_problem.py`](session-03_car_problem.py) | 🏙️ Real-life data problem: pricing used cars | `python session-03_car_problem.py` |
| [`session-03_mini_project.py`](session-03_mini_project.py) | 🧩 Mini project: house rent estimator | `python session-03_mini_project.py` |
| [`session-03_nonlinear_models.py`](session-03_nonlinear_models.py) | Example 4 · Advanced: trees, forests, boosting and SVR on a non-linear problem | `python session-03_nonlinear_models.py` |
| [`session-03_polynomial.py`](session-03_polynomial.py) | Example 1 · Basic: polynomial regression and overfitting | `python session-03_polynomial.py` |
| [`session-03_polynomial_fits.png`](session-03_polynomial_fits.png) | Example 1 · Basic: polynomial regression and overfitting | image produced by the matching script |
| [`session-03_reg_paths.py`](session-03_reg_paths.py) | Example 2 · Intermediate: Ridge and Lasso coefficient paths | `python session-03_reg_paths.py` |
| [`session-03_reg_rescue.py`](session-03_reg_rescue.py) | Example 3 · Intermediate: regularisation rescues an overfit model | `python session-03_reg_rescue.py` |
| [`session-03_regularisation_paths.png`](session-03_regularisation_paths.png) | Example 2 · Intermediate: Ridge and Lasso coefficient paths | image produced by the matching script |
| [`session-03_validation_curve.png`](session-03_validation_curve.png) | Example 5 · Advanced: validation curve for choosing alpha | image produced by the matching script |
| [`session-03_validation_curve.py`](session-03_validation_curve.py) | Example 5 · Advanced: validation curve for choosing alpha | `python session-03_validation_curve.py` |

### [Session 4 · Advanced Supervised Learning: Classification Techniques](../session-04-advanced-classification-techniques.md)

| File | What it does | How to run |
|---|---|---|
| [`session-04-advanced-classification-techniques.ipynb`](session-04-advanced-classification-techniques.ipynb) | Executed notebook with every Python example of the session | `jupyter notebook session-04-advanced-classification-techniques.ipynb` (or the Colab badge on the session page) |
| [`session-04_churn_importance.png`](session-04_churn_importance.png) | 🏙️ Real-life data problem: predicting telecom churn | image produced by the matching script |
| [`session-04_churn_problem.py`](session-04_churn_problem.py) | 🏙️ Real-life data problem: predicting telecom churn | `python session-04_churn_problem.py` |
| [`session-04_ensembles.py`](session-04_ensembles.py) | Example 3 · Intermediate: ensembles head to head | `python session-04_ensembles.py` |
| [`session-04_gini.py`](session-04_gini.py) | Example 1 · Basic: Gini and information gain by hand | `python session-04_gini.py` |
| [`session-04_mini_project.py`](session-04_mini_project.py) | 🧩 Mini project: loan approval classifier with tuned ensembles | `python session-04_mini_project.py` |
| [`session-04_svm_kernels.png`](session-04_svm_kernels.png) | Example 4 · Advanced: SVM margins, C and the kernel trick | image produced by the matching script |
| [`session-04_svm_kernels.py`](session-04_svm_kernels.py) | Example 4 · Advanced: SVM margins, C and the kernel trick | `python session-04_svm_kernels.py` |
| [`session-04_tree_depth.py`](session-04_tree_depth.py) | Example 2 · Intermediate: tree depth, pruning and the readable tree | `python session-04_tree_depth.py` |
| [`session-04_tuning.py`](session-04_tuning.py) | Example 5 · Advanced: grid search, random search, halving and nested CV | `python session-04_tuning.py` |

### [Session 5 · Advanced Supervised Learning: Time Series Analysis](../session-05-time-series-analysis.md)

| File | What it does | How to run |
|---|---|---|
| [`session-05-time-series-analysis.ipynb`](session-05-time-series-analysis.ipynb) | Executed notebook with every Python example of the session | `jupyter notebook session-05-time-series-analysis.ipynb` (or the Colab badge on the session page) |
| [`session-05_acf_pacf.png`](session-05_acf_pacf.png) | Example 2 · Intermediate: stationarity, differencing and ACF/PACF | image produced by the matching script |
| [`session-05_arima.py`](session-05_arima.py) | Example 3 · Intermediate: fitting ARIMA and choosing orders by AIC | `python session-05_arima.py` |
| [`session-05_decompose.py`](session-05_decompose.py) | Example 1 · Basic: building and decomposing a seasonal series | `python session-05_decompose.py` |
| [`session-05_decomposition.png`](session-05_decomposition.png) | Example 1 · Basic: building and decomposing a seasonal series | image produced by the matching script |
| [`session-05_forecast_comparison.png`](session-05_forecast_comparison.png) | Example 4 · Advanced: SARIMA vs Holt-Winters vs seasonal naive | image produced by the matching script |
| [`session-05_mini_project.py`](session-05_mini_project.py) | 🧩 Mini project: restaurant daily covers forecaster | `python session-05_mini_project.py` |
| [`session-05_power_forecast.png`](session-05_power_forecast.png) | 🏙️ Real-life data problem: monthly electricity demand for a distribution company | image produced by the matching script |
| [`session-05_power_problem.py`](session-05_power_problem.py) | 🏙️ Real-life data problem: monthly electricity demand for a distribution company | `python session-05_power_problem.py` |
| [`session-05_rolling_ml.py`](session-05_rolling_ml.py) | Example 5 · Advanced: rolling-origin evaluation and ML with lag features | `python session-05_rolling_ml.py` |
| [`session-05_seasonal_models.py`](session-05_seasonal_models.py) | Example 4 · Advanced: SARIMA vs Holt-Winters vs seasonal naive | `python session-05_seasonal_models.py` |
| [`session-05_stationarity.py`](session-05_stationarity.py) | Example 2 · Intermediate: stationarity, differencing and ACF/PACF | `python session-05_stationarity.py` |

### [Session 6 · Unsupervised Learning](../session-06-unsupervised-learning.md)

| File | What it does | How to run |
|---|---|---|
| [`session-06-unsupervised-learning.ipynb`](session-06-unsupervised-learning.ipynb) | Executed notebook with every Python example of the session | `jupyter notebook session-06-unsupervised-learning.ipynb` (or the Colab badge on the session page) |
| [`session-06_anomalies.py`](session-06_anomalies.py) | Example 5 · Advanced: anomaly detection, five ways | `python session-06_anomalies.py` |
| [`session-06_choose_k.png`](session-06_choose_k.png) | Example 2 · Intermediate: choosing k with elbow and silhouette | image produced by the matching script |
| [`session-06_choose_k.py`](session-06_choose_k.py) | Example 2 · Intermediate: choosing k with elbow and silhouette | `python session-06_choose_k.py` |
| [`session-06_cluster_families.png`](session-06_cluster_families.png) | Example 3 · Intermediate: four clustering families on tricky shapes | image produced by the matching script |
| [`session-06_cluster_families.py`](session-06_cluster_families.py) | Example 3 · Intermediate: four clustering families on tricky shapes | `python session-06_cluster_families.py` |
| [`session-06_customer_segments.png`](session-06_customer_segments.png) | 🏙️ Real-life data problem: segmenting supermarket customers | image produced by the matching script |
| [`session-06_kmeans_scratch.py`](session-06_kmeans_scratch.py) | Example 1 · Basic: K-means from scratch on 2-D data | `python session-06_kmeans_scratch.py` |
| [`session-06_mini_project.py`](session-06_mini_project.py) | 🧩 Mini project: unsupervised toolkit report for a city's air-quality sensors | `python session-06_mini_project.py` |
| [`session-06_pca_rules.py`](session-06_pca_rules.py) | Example 4 · Intermediate: PCA and association rules in a few lines | `python session-06_pca_rules.py` |
| [`session-06_rfm_problem.py`](session-06_rfm_problem.py) | 🏙️ Real-life data problem: segmenting supermarket customers | `python session-06_rfm_problem.py` |

### [Session 7 · Advanced Unsupervised Learning: Clustering Algorithms](../session-07-advanced-clustering-algorithms.md)

| File | What it does | How to run |
|---|---|---|
| [`session-07-advanced-clustering-algorithms.ipynb`](session-07-advanced-clustering-algorithms.ipynb) | Executed notebook with every Python example of the session | `jupyter notebook session-07-advanced-clustering-algorithms.ipynb` (or the Colab badge on the session page) |
| [`session-07_dbscan.png`](session-07_dbscan.png) | Example 3 · Intermediate: DBSCAN, the k-distance plot and parameter effects | image produced by the matching script |
| [`session-07_dbscan.py`](session-07_dbscan.py) | Example 3 · Intermediate: DBSCAN, the k-distance plot and parameter effects | `python session-07_dbscan.py` |
| [`session-07_dendrogram.png`](session-07_dendrogram.png) | Example 1 · Basic: agglomerative clustering and a dendrogram | image produced by the matching script |
| [`session-07_dendrogram.py`](session-07_dendrogram.py) | Example 1 · Basic: agglomerative clustering and a dendrogram | `python session-07_dendrogram.py` |
| [`session-07_evaluate_compare.py`](session-07_evaluate_compare.py) | Example 5 · Advanced: evaluation metrics and Mean Shift vs K-means vs DBSCAN | `python session-07_evaluate_compare.py` |
| [`session-07_gmm.py`](session-07_gmm.py) | Example 4 · Advanced: GMM vs K-means, soft membership and BIC | `python session-07_gmm.py` |
| [`session-07_gmm_vs_kmeans.png`](session-07_gmm_vs_kmeans.png) | Example 4 · Advanced: GMM vs K-means, soft membership and BIC | image produced by the matching script |
| [`session-07_hotspot_problem.py`](session-07_hotspot_problem.py) | 🏙️ Real-life data problem: finding ride-hailing pickup hotspots | `python session-07_hotspot_problem.py` |
| [`session-07_hotspots.png`](session-07_hotspots.png) | 🏙️ Real-life data problem: finding ride-hailing pickup hotspots | image produced by the matching script |
| [`session-07_linkages.py`](session-07_linkages.py) | Example 2 · Intermediate: comparing linkage methods | `python session-07_linkages.py` |
| [`session-07_mini_project.py`](session-07_mini_project.py) | 🧩 Mini project: segmenting an image's colours with K-means and Mean Shift | `python session-07_mini_project.py` |
| [`session-07_mini_project_segmentation.png`](session-07_mini_project_segmentation.png) | 🧩 Mini project: segmenting an image's colours with K-means and Mean Shift | image produced by the matching script |

### [Session 8 · Advanced Unsupervised Learning: Dimensionality Reduction](../session-08-dimensionality-reduction.md)

| File | What it does | How to run |
|---|---|---|
| [`session-08-dimensionality-reduction.ipynb`](session-08-dimensionality-reduction.ipynb) | Executed notebook with every Python example of the session | `jupyter notebook session-08-dimensionality-reduction.ipynb` (or the Colab badge on the session page) |
| [`session-08_mini_project.py`](session-08_mini_project.py) | 🧩 Mini project: compressing and visualising a product catalogue | `python session-08_mini_project.py` |
| [`session-08_mini_project_catalogue.png`](session-08_mini_project_catalogue.png) | 🧩 Mini project: compressing and visualising a product catalogue | image produced by the matching script |
| [`session-08_pca_digits.png`](session-08_pca_digits.png) | Example 2 · Intermediate: scree plot, choosing k and reconstruction | image produced by the matching script |
| [`session-08_pca_digits.py`](session-08_pca_digits.py) | Example 2 · Intermediate: scree plot, choosing k and reconstruction | `python session-08_pca_digits.py` |
| [`session-08_pca_pipeline.py`](session-08_pca_pipeline.py) | Example 3 · Intermediate: PCA inside a model pipeline | `python session-08_pca_pipeline.py` |
| [`session-08_pca_scratch.py`](session-08_pca_scratch.py) | Example 1 · Basic: PCA from scratch with NumPy | `python session-08_pca_scratch.py` |
| [`session-08_pca_vs_tsne.png`](session-08_pca_vs_tsne.png) | Example 4 · Advanced: PCA vs t-SNE on the same data | image produced by the matching script |
| [`session-08_pca_vs_tsne.py`](session-08_pca_vs_tsne.py) | Example 4 · Advanced: PCA vs t-SNE on the same data | `python session-08_pca_vs_tsne.py` |
| [`session-08_perplexity.png`](session-08_perplexity.png) | Example 5 · Advanced: the effect of perplexity (and seeds) | image produced by the matching script |
| [`session-08_perplexity.py`](session-08_perplexity.py) | Example 5 · Advanced: the effect of perplexity (and seeds) | `python session-08_perplexity.py` |
| [`session-08_survey_map.png`](session-08_survey_map.png) | 🏙️ Real-life data problem: understanding a 30-question customer survey | image produced by the matching script |
| [`session-08_survey_problem.py`](session-08_survey_problem.py) | 🏙️ Real-life data problem: understanding a 30-question customer survey | `python session-08_survey_problem.py` |

### [Session 9 · Pattern Mining and Visualization with Seaborn](../session-09-pattern-mining-and-seaborn.md)

| File | What it does | How to run |
|---|---|---|
| [`session-09-pattern-mining-and-seaborn.ipynb`](session-09-pattern-mining-and-seaborn.ipynb) | Executed notebook with every Python example of the session | `jupyter notebook session-09-pattern-mining-and-seaborn.ipynb` (or the Colab badge on the session page) |
| [`session-09_apriori_scratch.py`](session-09_apriori_scratch.py) | Example 1 · Basic: Apriori from scratch | `python session-09_apriori_scratch.py` |
| [`session-09_eclat.py`](session-09_eclat.py) | Example 2 · Intermediate: Eclat with tid-lists | `python session-09_eclat.py` |
| [`session-09_fptree.py`](session-09_fptree.py) | Example 3 · Intermediate: building an FP-Tree and mining with FP-Growth | `python session-09_fptree.py` |
| [`session-09_mba_problem.py`](session-09_mba_problem.py) | 🏙️ Real-life data problem: a supermarket's market basket analysis | `python session-09_mba_problem.py` |
| [`session-09_mba_rules.png`](session-09_mba_rules.png) | 🏙️ Real-life data problem: a supermarket's market basket analysis | image produced by the matching script |
| [`session-09_mini_project.py`](session-09_mini_project.py) | 🧩 Mini project: "frequently bought together" recommender | `python session-09_mini_project.py` |
| [`session-09_pairplot.png`](session-09_pairplot.png) | Example 5 · Advanced: more Seaborn for exploring patterns over time and groups | image produced by the matching script |
| [`session-09_rules_seaborn.png`](session-09_rules_seaborn.png) | Example 4 · Advanced: rule measures and Seaborn charts | image produced by the matching script |
| [`session-09_rules_seaborn.py`](session-09_rules_seaborn.py) | Example 4 · Advanced: rule measures and Seaborn charts | `python session-09_rules_seaborn.py` |
| [`session-09_seaborn_patterns.png`](session-09_seaborn_patterns.png) | Example 5 · Advanced: more Seaborn for exploring patterns over time and groups | image produced by the matching script |
| [`session-09_seaborn_patterns.py`](session-09_seaborn_patterns.py) | Example 5 · Advanced: more Seaborn for exploring patterns over time and groups | `python session-09_seaborn_patterns.py` |

### [Session 10 · Reinforcement Learning Basics](../session-10-reinforcement-learning-basics.md)

| File | What it does | How to run |
|---|---|---|
| [`session-10-reinforcement-learning-basics.ipynb`](session-10-reinforcement-learning-basics.ipynb) | Executed notebook with every Python example of the session | `jupyter notebook session-10-reinforcement-learning-basics.ipynb` (or the Colab badge on the session page) |
| [`session-10_bandit_regret.png`](session-10_bandit_regret.png) | Example 2 · Intermediate: four bandit strategies and regret | image produced by the matching script |
| [`session-10_bandits.py`](session-10_bandits.py) | Example 2 · Intermediate: four bandit strategies and regret | `python session-10_bandits.py` |
| [`session-10_cliff.py`](session-10_cliff.py) | Example 4 · Advanced: Monte Carlo, SARSA and Q-learning on Cliff Walking | `python session-10_cliff.py` |
| [`session-10_cliff_learning.png`](session-10_cliff_learning.png) | Example 4 · Advanced: Monte Carlo, SARSA and Q-learning on Cliff Walking | image produced by the matching script |
| [`session-10_mini_project.py`](session-10_mini_project.py) | 🧩 Mini project: a warehouse robot that learns to deliver | `python session-10_mini_project.py` |
| [`session-10_push_problem.py`](session-10_push_problem.py) | 🏙️ Real-life data problem: choosing the best push-notification message | `python session-10_push_problem.py` |
| [`session-10_push_traffic.png`](session-10_push_traffic.png) | 🏙️ Real-life data problem: choosing the best push-notification message | image produced by the matching script |
| [`session-10_reward_design.py`](session-10_reward_design.py) | Example 3 · Intermediate: reward design and reward hacking | `python session-10_reward_design.py` |
| [`session-10_rl_loop.py`](session-10_rl_loop.py) | Example 1 · Basic: the agent-environment loop and discounted return | `python session-10_rl_loop.py` |

### [Session 11 · Reinforcement Learning Principles](../session-11-reinforcement-learning-principles.md)

| File | What it does | How to run |
|---|---|---|
| [`session-11-reinforcement-learning-principles.ipynb`](session-11-reinforcement-learning-principles.ipynb) | Executed notebook with every Python example of the session | `jupyter notebook session-11-reinforcement-learning-principles.ipynb` (or the Colab badge on the session page) |
| [`session-11_bellman_hand.py`](session-11_bellman_hand.py) | Example 1 · Basic: a three-state MDP and the Bellman equation by hand | `python session-11_bellman_hand.py` |
| [`session-11_dp_grid.py`](session-11_dp_grid.py) | Example 2 · Intermediate: value iteration vs policy iteration on a stochastic grid | `python session-11_dp_grid.py` |
| [`session-11_exploration.png`](session-11_exploration.png) | Example 4 · Advanced: ε-greedy vs Softmax exploration | image produced by the matching script |
| [`session-11_exploration.py`](session-11_exploration.py) | Example 4 · Advanced: ε-greedy vs Softmax exploration | `python session-11_exploration.py` |
| [`session-11_mini_project.py`](session-11_mini_project.py) | 🧩 Mini project: when should a student study? (an MDP planner) | `python session-11_mini_project.py` |
| [`session-11_pharmacy_policy.png`](session-11_pharmacy_policy.png) | 🏙️ Real-life data problem: stock control for a pharmacy | image produced by the matching script |
| [`session-11_pharmacy_problem.py`](session-11_pharmacy_problem.py) | 🏙️ Real-life data problem: stock control for a pharmacy | `python session-11_pharmacy_problem.py` |
| [`session-11_qlearning_trace.py`](session-11_qlearning_trace.py) | Example 3 · Intermediate: Q-learning step by step with a TD-error trace | `python session-11_qlearning_trace.py` |

### [Session 12 · Deep Learning Models](../session-12-deep-learning-models.md)

| File | What it does | How to run |
|---|---|---|
| [`session-12-deep-learning-models.ipynb`](session-12-deep-learning-models.ipynb) | Executed notebook with every Python example of the session | `jupyter notebook session-12-deep-learning-models.ipynb` (or the Colab badge on the session page) |
| [`session-12_cnn_feature_maps.png`](session-12_cnn_feature_maps.png) | Example 3 · Intermediate: a CNN for handwritten digits, with feature maps | image produced by the matching script |
| [`session-12_mini_project.py`](session-12_mini_project.py) | 🧩 Mini project: activity recognition from phone sensors with a 1-D CNN | `python session-12_mini_project.py` |
| [`session-12_numpy_net.py`](session-12_numpy_net.py) | Example 1 · Basic: one neuron and a tiny network in NumPy (forward + backprop) | `python session-12_numpy_net.py` |
| [`session-12_reviews_problem.py`](session-12_reviews_problem.py) | 🏙️ Real-life data problem: classifying customer reviews (many-to-one text RNN) | `python session-12_reviews_problem.py` |
| [`session-12_torch_ann.py`](session-12_torch_ann.py) | Example 2 · Intermediate: an ANN in PyTorch with validation and early stopping | `python session-12_torch_ann.py` |
| [`session-12_torch_cnn.py`](session-12_torch_cnn.py) | Example 3 · Intermediate: a CNN for handwritten digits, with feature maps | `python session-12_torch_cnn.py` |
| [`session-12_torch_rnn.py`](session-12_torch_rnn.py) | Example 4 · Advanced: vanishing gradients (RNN vs LSTM) and an LSTM forecaster | `python session-12_torch_rnn.py` |

### [Session 13 · Advanced Deep Learning Techniques](../session-13-advanced-deep-learning.md)

| File | What it does | How to run |
|---|---|---|
| [`session-13-advanced-deep-learning.ipynb`](session-13-advanced-deep-learning.ipynb) | Executed notebook with every Python example of the session | `jupyter notebook session-13-advanced-deep-learning.ipynb` (or the Colab badge on the session page) |
| [`session-13_attention.py`](session-13_attention.py) | Example 4 · Intermediate: attention from scratch and multi-head attention in PyTorch | `python session-13_attention.py` |
| [`session-13_attention_heatmap.png`](session-13_attention_heatmap.png) | Example 4 · Intermediate: attention from scratch and multi-head attention in PyTorch | image produced by the matching script |
| [`session-13_dqn_thermostat.py`](session-13_dqn_thermostat.py) | 🏙️ Real-life data problem: a Deep Q-Network for a smart thermostat | `python session-13_dqn_thermostat.py` |
| [`session-13_gan_digits.py`](session-13_gan_digits.py) | Example 3 · Intermediate: a GAN that learns to draw digits | `python session-13_gan_digits.py` |
| [`session-13_gan_samples.png`](session-13_gan_samples.png) | Example 3 · Intermediate: a GAN that learns to draw digits | image produced by the matching script |
| [`session-13_gram_style.py`](session-13_gram_style.py) | Example 5 · Advanced: style transfer's key ingredient, the Gram matrix | `python session-13_gram_style.py` |
| [`session-13_mini_project.py`](session-13_mini_project.py) | 🧩 Mini project: a tiny transformer encoder classifier | `python session-13_mini_project.py` |
| [`session-13_resnet_block.py`](session-13_resnet_block.py) | Example 2 · Intermediate: a residual block and why depth needs it | `python session-13_resnet_block.py` |
| [`session-13_transfer_small.py`](session-13_transfer_small.py) | Example 1 · Basic: transfer learning with a small CNN (pre-train, then fine-tune) | `python session-13_transfer_small.py` |

### [Session 14 · Model Performance Evaluation](../session-14-model-performance-evaluation.md)

| File | What it does | How to run |
|---|---|---|
| [`session-14-model-performance-evaluation.ipynb`](session-14-model-performance-evaluation.ipynb) | Executed notebook with every Python example of the session | `jupyter notebook session-14-model-performance-evaluation.ipynb` (or the Colab badge on the session page) |
| [`session-14_bayes_opt.py`](session-14_bayes_opt.py) | Example 2 · Intermediate: Bayesian optimisation from scratch vs random search | `python session-14_bayes_opt.py` |
| [`session-14_curves.py`](session-14_curves.py) | Example 4 · Advanced: ROC, precision–recall and calibration curves | `python session-14_curves.py` |
| [`session-14_evaluation_curves.png`](session-14_evaluation_curves.png) | Example 4 · Advanced: ROC, precision–recall and calibration curves | image produced by the matching script |
| [`session-14_group_cv.py`](session-14_group_cv.py) | Example 5 · Advanced: the right cross-validation scheme (group leakage) | `python session-14_group_cv.py` |
| [`session-14_metrics_imbalanced.py`](session-14_metrics_imbalanced.py) | Example 3 · Intermediate: every classification metric on an imbalanced problem | `python session-14_metrics_imbalanced.py` |
| [`session-14_mini_project.py`](session-14_mini_project.py) | 🧩 Mini project: a reusable model evaluation report | `python session-14_mini_project.py` |
| [`session-14_optimisers.png`](session-14_optimisers.png) | Example 1 · Basic: optimisers on a difficult loss surface | image produced by the matching script |
| [`session-14_optimisers.py`](session-14_optimisers.py) | Example 1 · Basic: optimisers on a difficult loss surface | `python session-14_optimisers.py` |
| [`session-14_readmission_problem.py`](session-14_readmission_problem.py) | 🏙️ Real-life data problem: a hospital readmission model that the ward can trust | `python session-14_readmission_problem.py` |

### [Session 15 · Model Interpretability and Interoperability](../session-15-model-interpretability.md)

| File | What it does | How to run |
|---|---|---|
| [`session-15-model-interpretability.ipynb`](session-15-model-interpretability.ipynb) | Executed notebook with every Python example of the session | `jupyter notebook session-15-model-interpretability.ipynb` (or the Colab badge on the session page) |
| [`session-15_credit_problem.py`](session-15_credit_problem.py) | 🏙️ Real-life data problem: explaining credit decisions with reason codes | `python session-15_credit_problem.py` |
| [`session-15_lime_explain.py`](session-15_lime_explain.py) | Example 4 · Advanced: LIME for the same applicant, compared with SHAP | `python session-15_lime_explain.py` |
| [`session-15_mini_project.py`](session-15_mini_project.py) | 🧩 Mini project: an "explain my prediction" tool for house prices | `python session-15_mini_project.py` |
| [`session-15_onnx_export.py`](session-15_onnx_export.py) | Example 5 · Advanced: interoperability: export to ONNX and run without scikit-learn | `python session-15_onnx_export.py` |
| [`session-15_partial_dependence.png`](session-15_partial_dependence.png) | Example 2 · Intermediate: permutation importance and partial dependence | image produced by the matching script |
| [`session-15_perm_pdp.py`](session-15_perm_pdp.py) | Example 2 · Intermediate: permutation importance and partial dependence | `python session-15_perm_pdp.py` |
| [`session-15_shap_beeswarm.png`](session-15_shap_beeswarm.png) | Example 3 · Intermediate: SHAP with TreeExplainer (local and global) | image produced by the matching script |
| [`session-15_shap_tree.py`](session-15_shap_tree.py) | Example 3 · Intermediate: SHAP with TreeExplainer (local and global) | `python session-15_shap_tree.py` |
| [`session-15_shap_waterfall.png`](session-15_shap_waterfall.png) | Example 3 · Intermediate: SHAP with TreeExplainer (local and global) | image produced by the matching script |
| [`session-15_shapley_exact.py`](session-15_shapley_exact.py) | Example 1 · Basic: exact Shapley values by hand | `python session-15_shapley_exact.py` |

---

<sub>Prepared by **Hamna Zahid Khan**, Junior Lecturer</sub>
