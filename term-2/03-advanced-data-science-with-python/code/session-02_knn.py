"""KNN: a from-scratch classifier, then scikit-learn with scaling and a search for the best k."""
import numpy as np
from collections import Counter
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def knn_predict(X_train, y_train, x, k=3):
    """Return the majority label among the k nearest training points."""
    dist = np.sqrt(((X_train - x) ** 2).sum(axis=1))     # Euclidean distance to every point
    nearest = np.argsort(dist)[:k]
    return Counter(y_train[nearest]).most_common(1)[0][0]

# Tiny fruit example: [weight g, sweetness 1-10] -> 0 = lemon, 1 = orange
X_fruit = np.array([[120, 2], [130, 3], [110, 2], [150, 7], [170, 8], [160, 6]])
y_fruit = np.array([0, 0, 0, 1, 1, 1])
for fruit in ([125, 3], [165, 7], [140, 5]):
    label = knn_predict(X_fruit, y_fruit, np.array(fruit), k=3)
    print(f"fruit {fruit} -> {'orange' if label else 'lemon'}")

# Real data: breast tumour diagnosis, 30 features
X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)
raw = KNeighborsClassifier(5).fit(X_tr, y_tr).score(X_te, y_te)
scaled = make_pipeline(StandardScaler(), KNeighborsClassifier(5)).fit(X_tr, y_tr).score(X_te, y_te)
print(f"\nKNN k=5 accuracy without scaling: {raw:.3f}   with scaling: {scaled:.3f}")

print("\n k   5-fold CV accuracy")
for k in [1, 3, 5, 9, 15, 31, 75]:
    cv = cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier(k)), X_tr, y_tr, cv=5).mean()
    print(f"{k:>2}   {cv:.3f}")
