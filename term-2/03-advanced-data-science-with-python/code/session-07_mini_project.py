"""Mini project: colour segmentation of a synthetic satellite image with K-means and Mean Shift."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth

rng = np.random.default_rng(1)
H, W = 80, 120
yy, xx = np.mgrid[0:H, 0:W]
img = np.zeros((H, W, 3))
img[:] = [0.80, 0.70, 0.45]                                            # desert / bare soil
img[(xx - 30) ** 2 + (yy - 40) ** 2 < 400] = [0.15, 0.30, 0.65]       # lake
img[(yy > 50) & (xx > 60)] = [0.25, 0.55, 0.20]                       # farmland
img[(np.abs(yy - 0.4 * xx - 5) < 3)] = [0.45, 0.45, 0.45]             # road
img = np.clip(img + rng.normal(0, .05, img.shape), 0, 1)

pixels = img.reshape(-1, 3)
km = KMeans(4, n_init=10, random_state=0).fit(pixels)
seg_km = km.cluster_centers_[km.labels_].reshape(img.shape)
sample = pixels[rng.choice(len(pixels), 2000, replace=False)]
bw = estimate_bandwidth(sample, quantile=.1, random_state=0)
ms = MeanShift(bandwidth=bw, bin_seeding=True).fit(sample)
ms_labels = ms.predict(pixels)
seg_ms = ms.cluster_centers_[ms_labels].reshape(img.shape)

names = {0: "?", 1: "?", 2: "?", 3: "?"}
for j, c in enumerate(km.cluster_centers_):
    names[j] = "water" if c[2] > .5 else "vegetation" if c[1] > c[0] else "road" if abs(c[0] - c[1]) < .05 else "soil"
share = np.bincount(km.labels_) / len(pixels)
print("K-means land cover:", {names[j]: f"{share[j]:.1%}" for j in range(4)})
print(f"Mean Shift found {len(ms.cluster_centers_)} colour groups by itself (bandwidth {bw:.3f})")

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for a, im, t in zip(ax, [img, seg_km, seg_ms], ["noisy image", "K-means (k=4)", f"Mean Shift ({len(ms.cluster_centers_)} groups)"]):
    a.imshow(im); a.set_title(t); a.axis("off")
fig.tight_layout(); fig.savefig("session-07_mini_project_segmentation.png", dpi=110); plt.show()
