# 📝 Practice Test · Advanced Data Science with Python

> 📍 [Diploma in Data Science](../../README.md) › [Term 2](../README.md) › [Advanced Data Science with Python](README.md) › **Practice test**

Attempt this after finishing all 15 sessions. Suggested time: **60 minutes** (20 MCQs × 1 mark, 3 scenarios × 10 marks = 50 marks). Try every question before opening an answer.

[⬅ Back to Session 15](session-15-model-interpretability.md) · [🏠 Module home](README.md)

---

## Part A · Multiple-choice questions (20 marks)

**Q1.** Subtracting an array of shape (5,) from an array of shape (1000, 5) in NumPy works because of…

- **A)** vectorisation
- **B)** broadcasting
- **C)** boolean masking
- **D)** type casting

<details><summary>Show answer</summary>

**B** — NumPy stretches the smaller array across every row.

</details>

**Q2.** In gradient descent, a learning rate that is far too large usually makes the loss…

- **A)** fall slowly
- **B)** oscillate or explode
- **C)** stay exactly constant
- **D)** reach zero in one step

<details><summary>Show answer</summary>

**B** — Steps overshoot the minimum.

</details>

**Q3.** KNN performs badly on unscaled data because…

- **A)** it cannot handle numbers
- **B)** features with large units dominate the distance
- **C)** it needs labels
- **D)** it has no hyperparameters

<details><summary>Show answer</summary>

**B** — Distance-based methods need comparable scales.

</details>

**Q4.** Which regularisation can set some coefficients exactly to zero?

- **A)** Ridge (L2)
- **B)** Lasso (L1)
- **C)** dropout
- **D)** early stopping

<details><summary>Show answer</summary>

**B** — The L1 penalty produces sparse models.

</details>

**Q5.** A random forest reduces variance mainly by…

- **A)** boosting misclassified rows
- **B)** averaging many de-correlated trees built on bootstrap samples and random feature subsets
- **C)** pruning one deep tree
- **D)** scaling the features

<details><summary>Show answer</summary>

**B** — Bagging plus random feature selection.

</details>

**Q6.** In an RBF SVM, increasing gamma makes the decision boundary…

- **A)** smoother
- **B)** more local and wiggly
- **C)** linear
- **D)** disappear

<details><summary>Show answer</summary>

**B** — Each training point's influence becomes narrower.

</details>

**Q7.** A PACF that cuts off after lag 2 while the ACF tails off suggests…

- **A)** MA(2)
- **B)** AR(2)
- **C)** white noise
- **D)** seasonal differencing

<details><summary>Show answer</summary>

**B** — AR order is read from the PACF.

</details>

**Q8.** The correct way to evaluate a forecasting model is…

- **A)** a random train/test split
- **B)** training on the past and testing on later periods
- **C)** testing on the training data
- **D)** leave-one-out on random rows

<details><summary>Show answer</summary>

**B** — Time order must be respected.

</details>

**Q9.** Which clustering method can find curved clusters and label noise points?

- **A)** K-means
- **B)** DBSCAN
- **C)** Gaussian mixture
- **D)** Ward linkage

<details><summary>Show answer</summary>

**B** — DBSCAN is density-based.

</details>

**Q10.** For choosing the number of components in a Gaussian mixture, you typically use…

- **A)** silhouette only
- **B)** BIC
- **C)** the elbow of inertia
- **D)** lift

<details><summary>Show answer</summary>

**B** — Lower BIC balances fit and complexity.

</details>

**Q11.** On a t-SNE map, the relative sizes of clusters…

- **A)** show their true spread
- **B)** are not reliable
- **C)** equal their number of points
- **D)** show variance explained

<details><summary>Show answer</summary>

**B** — t-SNE equalises densities.

</details>

**Q12.** In PCA, the explained variance ratio of a component is…

- **A)** its eigenvalue divided by the sum of all eigenvalues
- **B)** its loading on the first feature
- **C)** the reconstruction error
- **D)** always 1 / number of features

<details><summary>Show answer</summary>

**A** — Components are ranked by this share.

</details>

**Q13.** A rule {tea} → {sugar} with lift 1.0 means…

- **A)** a strong association
- **B)** tea and sugar are independent
- **C)** sugar is never bought
- **D)** confidence is 100%

<details><summary>Show answer</summary>

**B** — Lift = 1 indicates independence.

</details>

**Q14.** Thompson sampling chooses an arm by…

- **A)** always picking the best average
- **B)** sampling from each arm's posterior and picking the highest sample
- **C)** choosing at random with probability ε
- **D)** the UCB formula

<details><summary>Show answer</summary>

**B** — It is a Bayesian exploration strategy.

</details>

**Q15.** SARSA differs from Q-learning because its update uses…

- **A)** the maximum next Q-value
- **B)** the Q-value of the next action actually taken
- **C)** the full episode return
- **D)** a neural network

<details><summary>Show answer</summary>

**B** — SARSA is on-policy.

</details>

**Q16.** Value iteration stops when…

- **A)** the policy is random
- **B)** the largest change in V between sweeps is below a small threshold
- **C)** ε reaches zero
- **D)** the reward is positive

<details><summary>Show answer</summary>

**B** — Values have converged.

</details>

**Q17.** The vanishing gradient problem mainly affects…

- **A)** decision trees
- **B)** plain RNNs on long sequences
- **C)** K-means
- **D)** linear regression

<details><summary>Show answer</summary>

**B** — Gradients shrink through many time steps; LSTMs/GRUs help.

</details>

**Q18.** In a GAN the discriminator's job is to…

- **A)** generate images
- **B)** tell real samples from generated ones
- **C)** compress images
- **D)** label classes

<details><summary>Show answer</summary>

**B** — It is the detective; the generator is the counterfeiter.

</details>

**Q19.** With 2% positives, which metric is most informative about the rare class?

- **A)** accuracy
- **B)** PR AUC
- **C)** R²
- **D)** MAE

<details><summary>Show answer</summary>

**B** — PR AUC focuses on precision and recall of the positive class.

</details>

**Q20.** SHAP values for a single prediction always sum to…

- **A)** 1
- **B)** the prediction minus the base value
- **C)** zero
- **D)** the number of features

<details><summary>Show answer</summary>

**B** — This is the additivity (local accuracy) property.

</details>


## Part B · Scenario questions (30 marks, 10 each)

### 🧩 Scenario 1 · Predicting electricity theft for a distribution company

A power distribution company in Karachi has monthly meter readings for 200,000 customers over three years, plus customer type (domestic/commercial), area, sanctioned load and the results of 6,000 past field inspections (1,100 confirmed thefts). Inspection teams can visit 2,000 premises per month. Design the modelling approach: data preparation, features, model choice, validation, metric and how you would choose whom to inspect. Mention at least two risks.

<details><summary>Show model solution</summary>

**Model answer (key points).**

- **Framing**: supervised binary classification (theft yes/no) trained on the 6,000 inspected customers, then used to score all 200,000. Positives are about 18% among inspected but much rarer overall; inspected customers are not a random sample (selection bias), so note it and, if possible, add randomly chosen inspections in future.
- **Features** (time-aware, using data up to the scoring month only): consumption trend and sudden drops (ratio of last 3 months to the same months last year), volatility, number of zero/estimated readings, consumption relative to sanctioned load and to similar customers in the same area and type (peer comparison), seasonality mismatch (no summer peak for AC users), payment behaviour, area theft rate (computed on training folds only to avoid leakage).
- **Models**: baseline logistic regression with scaled features; then gradient boosting (HistGradientBoosting); optionally an Isolation Forest anomaly score as an extra feature for uninspected patterns.
- **Validation**: time-based split (train on inspections up to month T, test on later inspections); GroupKFold by customer if a customer appears in several inspections; tune with RandomizedSearchCV on **PR AUC**.
- **Metric and decision**: capacity is 2,000 visits per month, so report **precision@2000** and recall at that capacity; rank customers by calibrated probability and inspect the top 2,000; compare the hit rate against the current selection method.
- **Explainability**: SHAP reason codes for each flagged customer ("consumption fell 60% vs last year while neighbours' did not") to guide inspectors and support legal follow-up.
- **Risks**: selection bias in past inspections; feedback loops (only flagged areas get inspected); fairness across areas/income groups; concept drift (new theft methods); legal/ethical issues of acting on model scores alone. Mitigate with random audit samples, monitoring and human review.

</details>


### 🧩 Scenario 2 · Segmenting and recommending for an online pharmacy

An online pharmacy has 18 months of orders: 90,000 customers, 400,000 orders and 3,000 products in 40 categories. Marketing wants customer segments for campaigns and a "frequently bought together" feature at checkout. Describe your unsupervised approach end to end, including how you would validate the results.

<details><summary>Show model solution</summary>

**Model answer (key points).**

- **Segmentation features** per customer: recency, frequency, monetary value (RFM), average basket size, share of spend by category group (chronic medicines, baby care, cosmetics, supplements), refill regularity (median days between repeat orders), discount usage, delivery city.
- **Preparation**: log-transform skewed RFM columns, standardise, optionally reduce correlated category shares with **PCA** (keep about 90% variance).
- **Clustering**: K-means for k = 3–10 with silhouette and elbow; compare with a GMM (BIC) for soft membership; profile each cluster (medians, top categories) and name it (e.g. "chronic refill regulars", "new parents", "occasional discount shoppers").
- **Validation**: stability across bootstrap samples and random seeds (ARI between runs), business review with marketing, and A/B tests of segment-specific campaigns measuring conversion and retention.
- **Recommendations**: build baskets per order at **category or product-group level** (SKU level is too sparse); exclude cancelled orders; mine with **FP-Growth** using min_support chosen so a few hundred itemsets appear; rules filtered by lift > 1.5 and confidence > 0.2; remove trivial or regulated combinations (prescription-only items need pharmacist rules).
- **Serving**: for a cart, rank consequents by lift × confidence of rules whose antecedent is in the cart; fall back to popular items in the same category.
- **Visualise** with a Seaborn lift heatmap and a support–confidence scatter; t-SNE map of customers coloured by segment for presentations (without reading cluster sizes from it).
- **Validate** recommendations offline (hit rate on held-out recent orders) and online (A/B test add-to-cart rate).
- **Ethics**: avoid exposing sensitive health inferences in marketing messages; follow privacy rules.

</details>


### 🧩 Scenario 3 · Dynamic discount policy for a food-delivery app

A food-delivery app wants to decide, for each restaurant and hour, whether to offer no discount, 10% or 20% to maximise long-run profit, considering that frequent discounts may train customers to wait for them. Explain how you would approach this with reinforcement learning, including how you would make the solution safe and explainable before deployment.

<details><summary>Show model solution</summary>

**Model answer (key points).**

- **Start simple**: treat each restaurant-hour as a **contextual bandit** (actions: 0%, 10%, 20%; context: hour, weekday, weather, restaurant type, recent demand) with **Thompson sampling**, reward = order profit after discount. This learns quickly and safely.
- **Why full RL may be needed**: discounts change future behaviour (customers wait for discounts), so today's action affects future states. Formulate an **MDP**: state = demand level, recent discount frequency, time features, customer price sensitivity estimate; actions = discount levels; reward = profit; γ around 0.95 (daily decisions).
- **Learning**: build a **simulator** from historical data (demand response to discounts estimated by regression or causal methods), solve with value iteration if discretised, or train **Q-learning/DQN** in the simulator; compare with baselines (never discount, always 10%, current manual policy).
- **Reward design**: profit net of discount cost, plus a penalty for very frequent discounting (to avoid training customers to wait); watch for reward hacking such as discounting only at times with naturally high demand.
- **Evaluation**: off-policy evaluation on logged data, then a limited A/B test in a few cities with guardrails (maximum discount budget per day, no more than N discounted hours per week).
- **Explainability**: summarise the learned policy as readable rules (a decision tree fitted to the agent's actions), report SHAP on the Q-network inputs or show Q-values per action so managers see why a discount was chosen.
- **Monitoring**: track profit, order volume, discount frequency and customer retention; retrain as behaviour drifts; keep a manual override.

</details>

---

<sub>Diploma in Data Science · Prepared by **Hamna Zahid Khan**, Junior Lecturer</sub>
