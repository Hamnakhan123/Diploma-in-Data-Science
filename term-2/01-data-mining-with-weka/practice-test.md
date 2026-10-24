# 📝 Practice Test · Data Mining with WEKA

> 📍 [Diploma in Data Science](../../README.md) › [Term 2](../README.md) › [Data Mining with WEKA](README.md) › **Practice test**

Attempt this after finishing all 8 sessions. Suggested time: **60 minutes** (20 MCQs × 1 mark, 3 scenarios × 10 marks = 50 marks). Try every question before opening an answer.

[⬅ Back to Session 8](session-08-real-world-data-mining-project.md) · [🏠 Module home](README.md)

---

## Part A · Multiple-choice questions (20 marks)

**Q1.** In the KDD process, data mining is…

- **A)** the first step
- **B)** one step, after selection, pre-processing and transformation
- **C)** the same as data warehousing
- **D)** only the evaluation step

<details><summary>Show answer</summary>

**B** — KDD is the whole process; data mining is the pattern-finding step inside it.

</details>

**Q2.** WEKA's native file format is…

- **A)** CSV
- **B)** JSON
- **C)** ARFF
- **D)** XLSX

<details><summary>Show answer</summary>

**C** — ARFF (Attribute-Relation File Format) stores the attribute types with the data.

</details>

**Q3.** A fact table in a star schema mainly holds…

- **A)** descriptive text
- **B)** measures and foreign keys to dimensions
- **C)** only dates
- **D)** user passwords

<details><summary>Show answer</summary>

**B** — Measures (sales, quantity) sit in the fact table; descriptions sit in dimension tables.

</details>

**Q4.** Min-max normalisation of 50 when min = 20 and max = 80 gives…

- **A)** 0.25
- **B)** 0.5
- **C)** 0.75
- **D)** 30

<details><summary>Show answer</summary>

**B** — (50 − 20) ÷ (80 − 20) = 0.5.

</details>

**Q5.** Which filter fills missing values with the mean or mode in WEKA?

- **A)** Discretize
- **B)** ReplaceMissingValues
- **C)** Normalize
- **D)** Remove

<details><summary>Show answer</summary>

**B** — ReplaceMissingValues uses the mean for numeric and the mode for nominal attributes.

</details>

**Q6.** Rule {bread} → {butter} has support 0.2 and support(bread) = 0.4. Its confidence is…

- **A)** 0.2
- **B)** 0.4
- **C)** 0.5
- **D)** 0.8

<details><summary>Show answer</summary>

**C** — 0.2 ÷ 0.4 = 0.5.

</details>

**Q7.** A lift of 0.7 means the items are…

- **A)** strongly associated
- **B)** independent
- **C)** bought together less often than expected
- **D)** always bought together

<details><summary>Show answer</summary>

**C** — Lift below 1 means a negative association.

</details>

**Q8.** The Apriori property says…

- **A)** every subset of a frequent itemset is also frequent
- **B)** every superset of a frequent itemset is frequent
- **C)** rules need lift above 2
- **D)** items must be numeric

<details><summary>Show answer</summary>

**A** — This lets Apriori prune candidates whose subsets are infrequent.

</details>

**Q9.** J48 chooses a split using…

- **A)** the attribute with the lowest support
- **B)** gain ratio
- **C)** random selection
- **D)** the largest number of values

<details><summary>Show answer</summary>

**B** — J48 (C4.5) uses gain ratio to avoid favouring many-valued attributes.

</details>

**Q10.** On data with 90% "no", a model with 90% accuracy…

- **A)** is excellent
- **B)** may be no better than ZeroR
- **C)** must be overfitted
- **D)** has 90% recall

<details><summary>Show answer</summary>

**B** — Always predicting "no" also gives 90%.

</details>

**Q11.** Recall for the positive class is…

- **A)** TP ÷ (TP + FP)
- **B)** TP ÷ (TP + FN)
- **C)** TN ÷ (TN + FP)
- **D)** (TP + TN) ÷ total

<details><summary>Show answer</summary>

**B** — Recall is the share of real positives the model finds.

</details>

**Q12.** Which WEKA scheme fits a tree with linear models in its leaves?

- **A)** REPTree
- **B)** J48
- **C)** M5P
- **D)** ZeroR

<details><summary>Show answer</summary>

**C** — M5P is a model tree.

</details>

**Q13.** RMSE is larger than MAE when…

- **A)** all errors are equal
- **B)** some errors are much larger than others
- **C)** the model is perfect
- **D)** the data has no outliers

<details><summary>Show answer</summary>

**B** — Squaring magnifies large errors, so uneven errors raise RMSE above MAE.

</details>

**Q14.** k-means requires you to…

- **A)** give the number of clusters
- **B)** provide class labels
- **C)** use nominal data only
- **D)** build a dendrogram

<details><summary>Show answer</summary>

**A** — k must be chosen in advance.

</details>

**Q15.** Which method can mark points as noise?

- **A)** k-means
- **B)** Ward hierarchical clustering
- **C)** DBSCAN
- **D)** Naive Bayes

<details><summary>Show answer</summary>

**C** — DBSCAN labels points outside dense regions as noise.

</details>

**Q16.** Silhouette scores range from…

- **A)** 0 to 1
- **B)** −1 to 1
- **C)** 0 to 100
- **D)** −∞ to ∞

<details><summary>Show answer</summary>

**B** — Values near 1 mean a point sits well inside its cluster.

</details>

**Q17.** In TF-IDF, a word in every document gets…

- **A)** the highest weight
- **B)** a weight near zero
- **C)** a negative weight
- **D)** a weight equal to its count

<details><summary>Show answer</summary>

**B** — IDF = log(N ÷ N) = 0.

</details>

**Q18.** Analysing clickstream logs is…

- **A)** web content mining
- **B)** web structure mining
- **C)** web usage mining
- **D)** text classification

<details><summary>Show answer</summary>

**C** — Usage mining studies how visitors move through a site.

</details>

**Q19.** Which CRISP-DM phase comes directly after data preparation?

- **A)** Business understanding
- **B)** Modelling
- **C)** Deployment
- **D)** Data understanding

<details><summary>Show answer</summary>

**B** — Business → Data understanding → Preparation → Modelling → Evaluation → Deployment.

</details>

**Q20.** Why use FilteredClassifier when a filter learns from data?

- **A)** It is the only classifier in WEKA
- **B)** The filter is fitted inside each training fold, avoiding leakage
- **C)** It removes the class
- **D)** It is required for ARFF

<details><summary>Show answer</summary>

**B** — Fitting the filter on all data would leak test information into training.

</details>


## Part B · Scenario questions (30 marks, 10 each)

### 🧩 Scenario 1 · Supermarket basket analysis

A Karachi supermarket gives you 50,000 receipts. Management wants to redesign shelf placement and plan bundle offers.

1. Which data mining technique would you use and why? (2)
2. Describe the pre-processing needed before running it in WEKA. (3)
3. Apriori returns {chips, soft drink} → {ice cream} with support 0.04, confidence 0.62 and lift 2.1, and {milk} → {bread} with support 0.20, confidence 0.55 and lift 0.98. Which rule is more useful for a bundle offer? Explain. (3)
4. Give one risk of acting on rules without further checks. (2)

<details><summary>Show model solution</summary>

1. **Association rule mining** (Apriori or FP-Growth), because the goal is to find items bought together, with no target variable.
2. Convert receipts into a **transaction × item** table with nominal or binary attributes (bought / not bought); remove returns, cancelled and test transactions; group very rare items or product variants into categories; choose minimum support and confidence (start low, e.g. 1% support, 50% confidence, then refine); FP-Growth is faster for 50,000 baskets.
3. **{chips, soft drink} → {ice cream}**. Lift 2.1 means customers who buy chips and a soft drink are about twice as likely as average to buy ice cream, a real association. The milk → bread rule has high support but **lift below 1**: bread is popular anyway and milk buyers are slightly *less* likely than average to buy it, so a bundle adds nothing.
4. Association is not causation: a rule may reflect a temporary promotion or season; it should be tested (for example an A/B trial in a few branches) before changing every store. Very low-support rules may also be chance findings.

</details>


### 🧩 Scenario 2 · Loan default prediction

A microfinance bank has 8,000 past loans, of which 6% defaulted. A junior analyst builds a J48 tree and reports 94% accuracy.

1. Why is the analyst's result not convincing? (2)
2. Which evaluation measures and method would you report instead? (3)
3. Suggest two ways to improve detection of defaulters in WEKA. (3)
4. One attribute is "number of late payments in the last 3 months of the loan". What is the problem? (2)

<details><summary>Show model solution</summary>

1. With 6% defaulters, **ZeroR** (always "no default") also scores 94%. The model may be catching no defaulters at all.
2. Report the **confusion matrix**, **recall and precision for the default class**, F1 and **ROC / PRC area**, using **stratified 10-fold cross-validation** (or a time-based hold-out if loans are ordered in time), and always compare with the ZeroR baseline.
3. Any two of: **CostSensitiveClassifier** with a higher cost for missed defaulters; **resampling** (SMOTE package, or `SpreadSubsample` / `ClassBalancer`) inside `FilteredClassifier`; trying stronger models such as **RandomForest**; lowering the decision **threshold** using the threshold curve; adding better features (income ratio, previous loans).
4. It is **data leakage**: late payments near the end of the loan are only known after the outcome is almost decided. At the time of approving a new loan this value does not exist, so the model would look excellent in testing and fail in real use.

</details>


### 🧩 Scenario 3 · Planning a student-retention project

A university wants to reduce first-year dropouts. It has admission records, attendance, quiz marks, fee payment dates and LMS login logs.

1. Write a business goal and a measurable success criterion. (2)
2. Map the project to the six CRISP-DM phases, with one concrete activity per phase. (4)
3. Which two techniques from this module would you use, and for what? (2)
4. Name two ethical safeguards. (2)

<details><summary>Show model solution</summary>

1. Goal: identify first-year students at risk of dropping out early enough for advisers to help. Success: by week 6, flag at least **75% of students who later drop out**, with at least half of flagged students genuinely at risk, so advisers' time is well used.
2. **Business understanding**: agree the goal, the definition of dropout and the adviser capacity with the student-affairs office. **Data understanding**: profile attendance, marks and LMS logins; check missing values and date ranges. **Data preparation**: join sources per student, derive features (attendance %, login count per week, late fee payment), handle missing values, use only data available by week 6. **Modelling**: compare J48, Logistic and RandomForest with stratified cross-validation, optimising recall. **Evaluation**: check the success criterion, review errors with advisers, test fairness across departments. **Deployment**: weekly risk list in the advisers' dashboard, monitoring of recall each semester, retraining every year.
3. **Classification** to predict dropout risk (with interpretable rules or a tree for advisers); **clustering** of students by engagement pattern to design different support (e.g. low attendance versus low marks). Web usage mining of LMS logs is also acceptable.
4. Any two of: exclude sensitive attributes (gender, religion, ethnicity) and check proxies; use predictions only to **offer support**, never to penalise; anonymise data for analysis and restrict access; explain to students how data is used and get approval from the university's ethics or data-protection body; a human adviser always makes the final decision.

</details>

---

<sub>Diploma in Data Science · Prepared by **Hamna Zahid Khan**, Junior Lecturer</sub>
