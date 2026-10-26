"""Mini project: a one-layer transformer encoder (self-attention) vs bag of words on messages where negation flips meaning."""
import numpy as np
import torch
from torch import nn
torch.manual_seed(0); torch.set_num_threads(2)
rng = np.random.default_rng(0)

subjects = ["please", "kindly", "i want to", "can you", "we need to", "urgent"]
items = ["my order", "the biryani", "order 4521", "the pizza", "this delivery", "my parcel"]
STOP, GO = ["cancel", "stop", "hold"], ["deliver", "send", "dispatch"]
def message():
    stop_verb = rng.random() < .5
    neg = rng.random() < .5
    verb = rng.choice(STOP if stop_verb else GO)
    text = f"{rng.choice(subjects)} {'do not ' if neg else ''}{verb} {rng.choice(items)}"
    # the customer wants the order STOPPED: 'cancel' or 'do not deliver' -> 1; 'deliver' or 'do not cancel' -> 0
    return text, int(stop_verb != neg)

data = [message() for _ in range(3000)]
vocab = {"<pad>": 0}
for t, _ in data:
    for w in t.split(): vocab.setdefault(w, len(vocab))
L = 10
enc = lambda t: ([vocab.get(w, 0) for w in t.split()] + [0] * L)[:L]
X = torch.tensor([enc(t) for t, _ in data]); Y = torch.tensor([y for _, y in data], dtype=torch.float32)
tr, te = slice(0, 2500), slice(2500, 3000)

class TinyTransformer(nn.Module):
    def __init__(self, V, d=32):
        super().__init__()
        self.emb = nn.Embedding(V, d, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(L, d) * .1)              # learned positional encoding
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=64, batch_first=True), 1)
        self.out = nn.Linear(d, 1)
    def forward(self, x):
        h = self.enc(self.emb(x) + self.pos, src_key_padding_mask=(x == 0))
        mask = (x != 0).unsqueeze(2).float()
        return self.out((h * mask).sum(1) / mask.sum(1)).squeeze(1)

class BagOfWords(nn.Module):                                          # baseline: ignores order and interactions
    def __init__(self, V, d=32):
        super().__init__(); self.emb = nn.EmbeddingBag(V, d, padding_idx=0); self.out = nn.Linear(d, 1)
    def forward(self, x): return self.out(self.emb(x)).squeeze(1)

for name, model in [("bag of words", BagOfWords(len(vocab))), ("tiny transformer", TinyTransformer(len(vocab)))]:
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for epoch in range(15):
        model.train(); perm = torch.randperm(2500)
        for i in range(0, 2500, 64):
            b = perm[i:i + 64]
            opt.zero_grad(); nn.functional.binary_cross_entropy_with_logits(model(X[b]), Y[b]).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        acc = ((model(X[te]) > 0).float() == Y[te]).float().mean().item()
    print(f"{name:<17} test accuracy: {acc:.3f}")

tests = ["please cancel my order", "please do not cancel my order", "do not deliver the pizza", "kindly send my parcel"]
with torch.no_grad():
    p = torch.sigmoid(model(torch.tensor([enc(t) for t in tests])))
for t, v in zip(tests, p):
    print(f"P(stop order) = {v:.2f}  '{t}'")
