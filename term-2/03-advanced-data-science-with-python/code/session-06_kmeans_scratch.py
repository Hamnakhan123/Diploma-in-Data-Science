"""K-means written from scratch with NumPy: assign, update, repeat, and track inertia."""
import numpy as np

rng = np.random.default_rng(0)
# Three neighbourhoods of delivery addresses (km east, km north)
X = np.vstack([rng.normal([2, 2], .6, (50, 2)), rng.normal([8, 3], .8, (50, 2)), rng.normal([5, 8], .7, (50, 2))])

def kmeans(X, k, iters=20, seed=1):
    r = np.random.default_rng(seed)
    centres = X[r.choice(len(X), k, replace=False)]           # random initial centres
    for it in range(iters):
        d = np.linalg.norm(X[:, None, :] - centres[None, :, :], axis=2)   # distance to each centre
        labels = d.argmin(axis=1)                                         # assign
        new = np.array([X[labels == j].mean(axis=0) for j in range(k)])   # update
        inertia = ((X - new[labels]) ** 2).sum()
        print(f"iteration {it + 1}: inertia = {inertia:8.2f}")
        if np.allclose(new, centres):
            break
        centres = new
    return labels, centres

labels, centres = kmeans(X, 3)
print("\nfinal centres (km):\n", centres.round(2))
print("points per cluster:", np.bincount(labels))
