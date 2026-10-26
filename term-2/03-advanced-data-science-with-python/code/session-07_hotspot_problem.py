"""Real-life problem: GPS pickup hotspots with DBSCAN (haversine metres), compared with K-means and GMM."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture

rng = np.random.default_rng(10)
EARTH_R = 6_371_000                                   # metres
hotspots = {"Saddar": (24.8607, 67.0104, 500), "Clifton": (24.8138, 67.0300, 450), "Gulshan": (24.9180, 67.0971, 600),
            "Airport": (24.8990, 67.1680, 350), "Tariq Road": (24.8720, 67.0600, 400), "DHA Ph 6": (24.7950, 67.0650, 300)}
rows = []
for name, (lat, lon, n) in hotspots.items():
    spread = rng.uniform(.0025, .0045)                # roughly 250-500 m
    rows.append(np.c_[rng.normal(lat, spread, n), rng.normal(lon, spread, n)])
stray = np.c_[rng.uniform(24.78, 24.95, 400), rng.uniform(66.98, 67.20, 400)]
pts = np.vstack(rows + [stray])
print("pickups:", len(pts))

# DBSCAN with haversine: inputs in radians, eps = metres / Earth radius
eps_m = 300
db = DBSCAN(eps=eps_m / EARTH_R, min_samples=25, metric="haversine", algorithm="ball_tree").fit(np.radians(pts))
lab = db.labels_
summary = (pd.DataFrame({"lat": pts[:, 0], "lon": pts[:, 1], "cluster": lab})
           .query("cluster != -1").groupby("cluster")
           .agg(pickups=("lat", "size"), centre_lat=("lat", "mean"), centre_lon=("lon", "mean"))
           .sort_values("pickups", ascending=False))
def nearest(lat, lon):
    return min(hotspots, key=lambda h: (hotspots[h][0] - lat) ** 2 + (hotspots[h][1] - lon) ** 2)
summary["area"] = [nearest(a, b) for a, b in zip(summary.centre_lat, summary.centre_lon)]
print(f"\nDBSCAN (eps={eps_m} m): {len(summary)} hotspots, {(lab == -1).sum()} pickups treated as stray")
print(summary.round(4).to_string())

km = KMeans(6, n_init=10, random_state=0).fit(pts)
gmm = GaussianMixture(6, random_state=0).fit(pts)
stray_mask = np.arange(len(pts)) >= len(pts) - 400
print(f"\nstray pickups absorbed into a 'hotspot' -> K-means: {stray_mask.sum()} of 400 (every point is assigned)")
print(f"stray pickups correctly left out by DBSCAN: {(lab[stray_mask] == -1).sum()} of 400")
print(f"GMM: stray pickups with low likelihood (bottom 15%): "
      f"{(gmm.score_samples(pts)[stray_mask] < np.percentile(gmm.score_samples(pts), 15)).sum()} of 400")

plt.figure(figsize=(7, 6))
plt.scatter(pts[lab == -1, 1], pts[lab == -1, 0], s=4, color="lightgrey", label="stray")
plt.scatter(pts[lab != -1, 1], pts[lab != -1, 0], c=lab[lab != -1], cmap="plasma", s=5)
for _, r in summary.iterrows():
    plt.annotate(f"{r.area} ({r.pickups})", (r.centre_lon, r.centre_lat), fontsize=8, weight="bold")
plt.xlabel("longitude"); plt.ylabel("latitude"); plt.title("Evening pickup hotspots found by DBSCAN"); plt.legend()
plt.tight_layout(); plt.savefig("session-07_hotspots.png", dpi=110); plt.show()
