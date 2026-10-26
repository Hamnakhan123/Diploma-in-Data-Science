"""Hierarchical clustering of 12 cities by climate with SciPy: linkage matrix, dendrogram, cutting the tree."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler

cities = pd.DataFrame({
    "city": ["Karachi", "Hyderabad", "Sukkur", "Lahore", "Faisalabad", "Multan",
             "Islamabad", "Peshawar", "Murree", "Quetta", "Gilgit", "Gwadar"],
    "avg_temp_c": [26.5, 27.5, 27.0, 24.5, 24.5, 26.0, 21.5, 23.0, 13.5, 15.5, 16.0, 26.5],
    "rain_mm":    [175, 160, 95, 630, 380, 190, 1140, 460, 1700, 250, 140, 100],
    "humidity":   [70, 55, 45, 55, 55, 50, 58, 52, 70, 35, 40, 72],
})
X = StandardScaler().fit_transform(cities[["avg_temp_c", "rain_mm", "humidity"]])
Z = linkage(X, method="ward")                       # each row: cluster a, cluster b, distance, size
print("first 4 merges (a, b, distance, size):\n", np.round(Z[:4], 2))
cities["group"] = fcluster(Z, t=4, criterion="maxclust")
for g, names in cities.groupby("group")["city"]:
    print(f"group {g}: {', '.join(names)}")

plt.figure(figsize=(9, 4))
dendrogram(Z, labels=cities["city"].tolist(), leaf_rotation=45, color_threshold=Z[-4, 2])
plt.axhline(Z[-4, 2], ls="--", color="grey"); plt.ylabel("Ward merge distance")
plt.title("Dendrogram of cities by climate (dashed line = cut into 4 groups)")
plt.tight_layout(); plt.savefig("session-07_dendrogram.png", dpi=110); plt.show()
