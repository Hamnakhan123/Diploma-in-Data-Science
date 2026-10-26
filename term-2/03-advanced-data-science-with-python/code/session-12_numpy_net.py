"""XOR with a 2-4-1 network written in NumPy: forward pass, backpropagation and gradient descent."""
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)          # XOR: not linearly separable
rng = np.random.default_rng(1)
W1, b1 = rng.normal(0, 1, (2, 4)), np.zeros((1, 4))
W2, b2 = rng.normal(0, 1, (4, 1)), np.zeros((1, 1))
sigmoid = lambda z: 1 / (1 + np.exp(-z))

for epoch in range(5001):
    # forward pass
    h = np.tanh(X @ W1 + b1)                 # hidden layer (4 neurons)
    p = sigmoid(h @ W2 + b2)                 # output probability
    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    # backward pass (chain rule)
    dz2 = (p - y) / len(X)                   # dLoss/dz for sigmoid + cross-entropy
    dW2, db2 = h.T @ dz2, dz2.sum(0, keepdims=True)
    dh = dz2 @ W2.T * (1 - h ** 2)           # back through tanh
    dW1, db1 = X.T @ dh, dh.sum(0, keepdims=True)
    # gradient descent step
    for P, G in [(W1, dW1), (b1, db1), (W2, dW2), (b2, db2)]:
        P -= 1.0 * G
    if epoch in (0, 100, 1000, 5000):
        print(f"epoch {epoch:>4}: loss = {loss:.4f}  predictions = {p.ravel().round(3)}")
print("rounded predictions:", (p.ravel() > .5).astype(int), " true:", y.ravel().astype(int))
