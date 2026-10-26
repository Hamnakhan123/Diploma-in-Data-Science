"""Transfer learning in miniature: pre-train on clean digit scans, reuse for noisy 'phone photos' with 5 labels per class."""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
torch.set_num_threads(2)

d = load_digits()
X = F.interpolate(torch.tensor(d.images / 16.0, dtype=torch.float32).unsqueeze(1), size=16, mode="bilinear", align_corners=False)
y = torch.tensor(d.target)
rng = np.random.default_rng(0)
idx = rng.permutation(len(y))
src, tgt = idx[:900], idx[900:]                                    # different images for the two tasks
# Target domain = 'phone photos': lower contrast, background noise and random shifts
gen = torch.Generator().manual_seed(0)
Xt = 0.6 * X + 0.35 * torch.rand(X.shape, generator=gen)
Xt = torch.stack([torch.roll(img, shifts=(int(rng.integers(-2, 3)), int(rng.integers(-2, 3))), dims=(1, 2)) for img in Xt])
few = np.concatenate([rng.choice(tgt[d.target[tgt] == c], 5, replace=False) for c in range(10)])   # only 50 labels
test = np.setdiff1d(tgt, few)

def backbone():
    return nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten())

def train(model, data, idx, epochs, lr, params=None):
    opt = torch.optim.Adam(params if params is not None else model.parameters(), lr=lr)
    for ep in range(epochs):
        perm = rng.permutation(idx)
        for i in range(0, len(perm), 32):
            b = perm[i:i + 32]
            opt.zero_grad(); F.cross_entropy(model(data[b]), y[b]).backward(); opt.step()

def acc(model, data, idx):
    model.eval()
    with torch.no_grad():
        return (model(data[idx]).argmax(1) == y[idx]).float().mean().item()

torch.manual_seed(0)
pre = nn.Sequential(backbone(), nn.Linear(512, 10))
train(pre, X, src, epochs=15, lr=2e-3)                               # source: 900 clean labelled scans
print(f"pre-trained model, clean scans (held out): {acc(pre, X, tgt):.3f}")
print(f"pre-trained model used as-is on noisy photos: {acc(pre, Xt, test):.3f}")

results = {}
torch.manual_seed(1)
scratch = nn.Sequential(backbone(), nn.Linear(512, 10)); train(scratch, Xt, few, 80, 2e-3)
results["from scratch (50 photos)"] = acc(scratch, Xt, test)
torch.manual_seed(1)
fe = nn.Sequential(backbone(), nn.Linear(512, 10)); fe.load_state_dict(pre.state_dict())
for p in fe[0].parameters(): p.requires_grad = False                 # freeze the backbone, retrain the head
train(fe, Xt, few, 80, 2e-3, params=fe[1].parameters())
results["feature extraction (frozen)"] = acc(fe, Xt, test)
torch.manual_seed(1)
ft = nn.Sequential(backbone(), nn.Linear(512, 10)); ft.load_state_dict(pre.state_dict())
train(ft, Xt, few, 80, 3e-4)                                         # small learning rate for all layers
results["fine-tuning (all layers)"] = acc(ft, Xt, test)
for k, v in results.items():
    print(f"{k:<30} accuracy on unseen noisy photos: {v:.3f}")
