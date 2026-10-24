import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

df = pd.DataFrame([
 (22,40,85),(25,45,90),(23,38,80),(27,50,88),(24,42,82),(45,180,20),(50,200,15),(48,190,25),(52,210,18),
 (47,170,22),(35,120,75),(38,130,80),(33,115,70),(36,125,78),(60,60,30),(58,55,35),(62,65,28),(59,50,32)],
 columns=["age", "income_k", "spend_score"])
X = StandardScaler().fit_transform(df)

print(" k      SSE  silhouette")
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    print(f"{k:>2} {km.inertia_:>8.2f} {silhouette_score(X, km.labels_):>10.3f}")

km = KMeans(n_clusters=4, n_init=10, random_state=0).fit(X)
df["kmeans"] = km.labels_
print("\nCluster profiles (means in original units):")
print(df.groupby("kmeans")[["age", "income_k", "spend_score"]].mean().round(1).assign(size=df["kmeans"].value_counts().sort_index()))

agg = AgglomerativeClustering(n_clusters=4, linkage="ward").fit(X)
same = pd.crosstab(df["kmeans"], agg.labels_)
print("\nk-means vs Ward agreement (each row should have one non-zero cell):")
print(same)

db = DBSCAN(eps=0.6, min_samples=3).fit(X)
print("\nDBSCAN labels (-1 = noise):", [int(l) for l in db.labels_])
