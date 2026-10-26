"""A small CNN on 8x8 digit images (upscaled to 16x16): training, test accuracy, filters and feature maps."""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
torch.manual_seed(0); torch.set_num_threads(2)

digits = load_digits()
X = torch.tensor(digits.images / 16.0, dtype=torch.float32).unsqueeze(1)      # (n, 1, 8, 8)
X = F.interpolate(X, size=16, mode="bilinear", align_corners=False)            # 16x16 for two pooling layers
y = torch.tensor(digits.target)
idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=.25, random_state=0, stratify=digits.target)

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)      # 16x16 -> 8 maps of 16x16
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)     # after pool 8x8 -> 16 maps of 8x8
        self.fc1 = nn.Linear(16 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)
        self.drop = nn.Dropout(0.3)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)      # -> 8 x 8 x 8
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)      # -> 16 x 4 x 4
        x = self.drop(F.relu(self.fc1(x.flatten(1))))
        return self.fc2(x)

model = SmallCNN()
print("parameters:", sum(p.numel() for p in model.parameters()))
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
for epoch in range(15):
    model.train()
    perm = torch.tensor(np.random.default_rng(epoch).permutation(idx_tr))
    for i in range(0, len(perm), 64):
        b = perm[i:i + 64]
        opt.zero_grad(); loss = F.cross_entropy(model(X[b]), y[b]); loss.backward(); opt.step()
    if epoch % 5 == 4:
        model.eval()
        with torch.no_grad():
            acc = (model(X[idx_te]).argmax(1) == y[idx_te]).float().mean().item()
        print(f"epoch {epoch + 1:>2}: last batch loss {loss.item():.3f}  test accuracy {acc:.3f}")

model.eval()
with torch.no_grad():
    maps = F.relu(model.conv1(X[idx_te[:1]]))[0]
fig, ax = plt.subplots(2, 9, figsize=(13, 3.4))
ax[0, 0].imshow(X[idx_te[0], 0], cmap="gray_r"); ax[0, 0].set_title(f"input: {y[idx_te[0]].item()}")
ax[1, 0].axis("off")
for k in range(8):
    ax[0, k + 1].imshow(model.conv1.weight[k, 0].detach(), cmap="RdBu"); ax[0, k + 1].set_title(f"filter {k + 1}", fontsize=8)
    ax[1, k + 1].imshow(maps[k], cmap="magma"); ax[1, k + 1].set_title(f"feature map {k + 1}", fontsize=8)
for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
fig.suptitle("Learned 3x3 filters (top) and the feature maps they produce for one digit (bottom)")
fig.tight_layout(); fig.savefig("session-12_cnn_feature_maps.png", dpi=100); plt.show()
