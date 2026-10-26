"""Real-life problem: RFM customer segmentation with K-means, profiling, PCA view and anomaly flags."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

rng = np.random.default_rng(12)
groups = [  # (count, recency days, visits/year, avg basket Rs)
    (600, 10, 60, 1800),    # loyal regulars
    (500, 40, 12, 4500),    # monthly big-basket shoppers
    (700, 120, 4, 900),     # occasional
    (200, 15, 35, 9000),    # high-value families
]
rows = []
for n, r, f, m in groups:
    rows.append(pd.DataFrame({"recency_days": rng.gamma(2, r / 2, n).round() + 1,
                              "frequency": rng.poisson(f, n) + 1,
                              "avg_basket": rng.lognormal(np.log(m), .25, n).round()}))
cust = pd.concat(rows, ignore_index=True)
cust["monetary"] = cust["frequency"] * cust["avg_basket"]
cust.loc[rng.choice(len(cust), 6, replace=False), "monetary"] *= 15     # a few suspicious accounts

features = ["recency_days", "frequency", "avg_basket", "monetary"]
X = StandardScaler().fit_transform(np.log1p(cust[features]))           # log first: RFM is very skewed
for k in range(2, 8):
    lab = KMeans(k, n_init=10, random_state=0).fit_predict(X)
    print(f"k={k}: silhouette = {silhouette_score(X, lab):.3f}")
km = KMeans(4, n_init=10, random_state=0).fit(X)
cust["segment"] = km.labels_

profile = cust.groupby("segment")[features].median().round(0)
profile["customers"] = cust["segment"].value_counts().sort_index()
profile = profile.sort_values("monetary", ascending=False)
names = ["Champions (high value)", "Loyal regulars", "Monthly stock-up", "Occasional / at risk"]
profile["name"] = names
print("\nSegment profiles (medians):\n", profile.to_string())

# Consistency check: total spend should equal visits x average basket; a big gap is suspicious
cust["spend_gap"] = np.log(cust["monetary"] / (cust["frequency"] * cust["avg_basket"]))
X_anom = np.c_[X, StandardScaler().fit_transform(cust[["spend_gap"]])]
cust["anomaly"] = IsolationForest(contamination=0.003, random_state=0).fit_predict(X_anom) == -1
print(f"\naccounts flagged for review: {cust['anomaly'].sum()}")
print(cust[cust["anomaly"]][features + ["spend_gap"]].round(1).to_string())

name_of = dict(zip(profile.index, names))
P = PCA(2).fit_transform(X)
plt.figure(figsize=(7, 5))
for s in profile.index:
    m = cust["segment"] == s
    plt.scatter(P[m, 0], P[m, 1], s=8, alpha=.6, label=name_of[s])
plt.scatter(P[cust.anomaly, 0], P[cust.anomaly, 1], marker="x", s=60, color="black", label="flagged anomaly")
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("Customer segments (PCA view of RFM features)"); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig("session-06_customer_segments.png", dpi=110); plt.show()
