"""Mini project: cluster air-quality stations, reduce dimensions with PCA and flag faulty readings."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

rng = np.random.default_rng(4)
profiles = {"industrial": [180, 90, 60, 45], "traffic": [140, 110, 35, 30], "residential": [80, 45, 20, 15], "coastal": [55, 25, 12, 10]}
rows = []
for zone, (pm25, no2, so2, co) in profiles.items():
    for s in range(12):
        for day in range(30):
            rows.append([f"{zone[:3].upper()}-{s:02d}", zone, *(np.array([pm25, no2, so2, co]) * rng.lognormal(0, .15, 4))])
df = pd.DataFrame(rows, columns=["station", "zone", "pm25", "no2", "so2", "co"])
df.loc[rng.choice(len(df), 15, replace=False), ["pm25", "co"]] = [[900, 2]]      # sensor glitches

pollutants = ["pm25", "no2", "so2", "co"]
X = StandardScaler().fit_transform(np.log(df[pollutants]))
df["faulty"] = IsolationForest(contamination=0.01, random_state=0).fit_predict(X) == -1
print("readings flagged as faulty:", int(df.faulty.sum()))

stations = df[~df.faulty].groupby("station")[pollutants].mean()
S = StandardScaler().fit_transform(np.log(stations))
best_k = max(range(2, 7), key=lambda k: silhouette_score(S, KMeans(k, n_init=10, random_state=0).fit_predict(S)))
stations["cluster"] = KMeans(best_k, n_init=10, random_state=0).fit_predict(S)
print("best number of station groups:", best_k)
print(stations.groupby("cluster")[pollutants].mean().round(1).assign(stations=stations.cluster.value_counts().sort_index()).to_string())

pca = PCA(2).fit(S)
print("\nPCA: 2 components explain", f"{pca.explained_variance_ratio_.sum():.1%}", "of station-level variation")
print("PC1 loadings:", {p: round(float(v), 2) for p, v in zip(pollutants, pca.components_[0])})
