"""Scaled dot-product attention in NumPy on a toy sentence, then PyTorch MultiheadAttention shapes."""
import numpy as np
import torch
import matplotlib.pyplot as plt

words = ["the", "rider", "delivered", "the", "biryani", "because", "he", "was", "fast"]
rng = np.random.default_rng(0)
d = 8
# Hand-made embeddings: 'he' is similar to 'rider'; 'fast' is similar to 'delivered'
E = rng.normal(0, .3, (len(words), d))
E[1] += [2, 0, 0, 0, 0, 0, 0, 0]; E[6] += [2, 0, 0, 0, 0, 0, 0, 0]          # rider ~ he
E[2] += [0, 2, 0, 0, 0, 0, 0, 0]; E[8] += [0, 2, 0, 0, 0, 0, 0, 0]          # delivered ~ fast
E[4] += [0, 0, 2, 0, 0, 0, 0, 0]

def attention(Q, K, V):
    scores = Q @ K.T / np.sqrt(Q.shape[1])
    w = np.exp(scores - scores.max(axis=1, keepdims=True)); w /= w.sum(axis=1, keepdims=True)
    return w @ V, w

out, W = attention(E, E, E)                       # self-attention with identity projections
for i in [6, 8]:
    top = np.argsort(-W[i])[:3]
    print(f"'{words[i]}' attends most to: " + ", ".join(f"{words[j]} ({W[i, j]:.2f})" for j in top))
print("each row of weights sums to 1:", np.allclose(W.sum(1), 1))

mha = torch.nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)
x = torch.tensor(E, dtype=torch.float32).unsqueeze(0)          # (batch 1, 9 tokens, 8 dims)
y, w = mha(x, x, x)
print("PyTorch multi-head output:", tuple(y.shape), "| averaged attention weights:", tuple(w.shape))

plt.figure(figsize=(6, 5))
plt.imshow(W, cmap="magma"); plt.colorbar(label="attention weight")
plt.xticks(range(len(words)), words, rotation=45); plt.yticks(range(len(words)), words)
plt.title("Self-attention weights (row = query word)")
plt.tight_layout(); plt.savefig("session-13_attention_heatmap.png", dpi=110); plt.show()
