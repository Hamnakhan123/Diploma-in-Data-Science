"""Mini project: classify walking, running, sitting and climbing stairs from simulated accelerometer windows (1-D CNN)."""
import numpy as np
import torch
from torch import nn
torch.manual_seed(0); torch.set_num_threads(2)
rng = np.random.default_rng(0)

activities = ["sitting", "walking", "running", "stairs"]
def window(label, n=64):
    t = np.arange(n) / 32.0                                    # 2 seconds at 32 Hz
    if label == 0:
        sig = np.c_[0.02 * rng.normal(size=n), 0.02 * rng.normal(size=n), 1 + 0.02 * rng.normal(size=n)]
    else:
        f, amp = {1: (1.8, .35), 2: (2.6, .8), 3: (1.6, .4)}[label]
        ph = rng.uniform(0, 2 * np.pi)
        z = 1 + amp * np.sin(2 * np.pi * f * t + ph)
        x = .5 * amp * np.sin(2 * np.pi * f / 2 * t + ph)
        y = (.3 if label == 3 else .1) * np.sin(2 * np.pi * f * t + ph + 1)    # stairs: more sideways sway
        sig = np.c_[x, y, z] + 0.35 * rng.normal(size=(n, 3))          # noisy phone sensor
    return sig.T                                               # (3 channels, 64 time steps)

y = rng.integers(0, 4, 2400)
X = torch.tensor(np.stack([window(l) for l in y]), dtype=torch.float32)
Y = torch.tensor(y)
tr, te = np.arange(2000), np.arange(2000, 2400)

model = nn.Sequential(nn.Conv1d(3, 16, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
                      nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
                      nn.Flatten(), nn.Linear(32, 4))
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
for epoch in range(10):
    perm = torch.tensor(rng.permutation(tr))
    for i in range(0, len(perm), 64):
        b = perm[i:i + 64]
        opt.zero_grad(); loss = nn.functional.cross_entropy(model(X[b]), Y[b]); loss.backward(); opt.step()
model.eval()
with torch.no_grad():
    pred = model(X[te]).argmax(1)
print(f"test accuracy: {(pred == Y[te]).float().mean().item():.3f}")
conf = np.zeros((4, 4), int)
for t_, p_ in zip(Y[te].numpy(), pred.numpy()):
    conf[t_, p_] += 1
print("confusion matrix (rows = true, columns = predicted):")
print("          " + " ".join(f"{a[:7]:>8}" for a in activities))
for a, row in zip(activities, conf):
    print(f"{a:<10}" + " ".join(f"{v:>8}" for v in row))
