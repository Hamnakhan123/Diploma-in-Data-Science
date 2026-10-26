"""Supervised, unsupervised and reinforcement learning side by side on tiny problems."""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

X, y = load_iris(return_X_y=True)

# 1) Supervised: labels are given, we learn to predict them
clf = LogisticRegression(max_iter=500).fit(X, y)
print("SUPERVISED   training accuracy:", round(clf.score(X, y), 3))

# 2) Unsupervised: labels hidden, K-means finds 3 groups on its own
km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
print("UNSUPERVISED agreement of clusters with true species (ARI):", round(adjusted_rand_score(y, km.labels_), 3))

# 3) Reinforcement: a 3-armed bandit; the agent only sees rewards
rng = np.random.default_rng(0)
true_win_rate = np.array([0.2, 0.5, 0.8])          # unknown to the agent
Q, N, total = np.zeros(3), np.zeros(3), 0
for t in range(1000):
    a = rng.integers(3) if rng.random() < 0.1 else int(np.argmax(Q))   # epsilon-greedy
    r = float(rng.random() < true_win_rate[a])
    N[a] += 1; Q[a] += (r - Q[a]) / N[a]; total += r
print("REINFORCEMENT estimated win rates:", Q.round(2), "| times each arm pulled:", N.astype(int))
print("              average reward:", total / 1000)
