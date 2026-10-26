"""A small fully connected GAN trained on 8x8 digit images; saves a grid of generated samples over training."""
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
torch.manual_seed(0); torch.set_num_threads(2)

real = torch.tensor(load_digits().images.reshape(-1, 64) / 8.0 - 1.0, dtype=torch.float32)   # scale to [-1, 1]
Z = 16
G = nn.Sequential(nn.Linear(Z, 128), nn.LeakyReLU(0.2), nn.Linear(128, 128), nn.LeakyReLU(0.2), nn.Linear(128, 64), nn.Tanh())
D = nn.Sequential(nn.Linear(64, 128), nn.LeakyReLU(0.2), nn.Dropout(0.3), nn.Linear(128, 1))
optG = torch.optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.999))
optD = torch.optim.Adam(D.parameters(), lr=1e-3, betas=(0.5, 0.999))
bce = nn.BCEWithLogitsLoss()
fixed_z = torch.randn(10, Z)
snapshots = {}
for step in range(3001):
    xb = real[torch.randint(0, len(real), (64,))]
    fake = G(torch.randn(64, Z))
    # 1) discriminator: real -> 1 (smoothed to 0.9), fake -> 0
    optD.zero_grad()
    lossD = bce(D(xb), torch.full((64, 1), 0.9)) + bce(D(fake.detach()), torch.zeros(64, 1))
    lossD.backward(); optD.step()
    # 2) generator: wants D to say 'real' for fakes
    optG.zero_grad()
    lossG = bce(D(fake), torch.ones(64, 1))
    lossG.backward(); optG.step()
    if step in (0, 300, 1000, 3000):
        with torch.no_grad():
            snapshots[step] = G(fixed_z).numpy()
            dr, df = torch.sigmoid(D(xb)).mean().item(), torch.sigmoid(D(fake)).mean().item()
        print(f"step {step:>4}: loss D {lossD.item():.3f}  loss G {lossG.item():.3f}  D(real) {dr:.2f}  D(fake) {df:.2f}")

fig, ax = plt.subplots(4, 10, figsize=(10, 4.4))
for r, (step, imgs) in enumerate(snapshots.items()):
    for c in range(10):
        ax[r, c].imshow(imgs[c].reshape(8, 8), cmap="gray_r"); ax[r, c].axis("off")
    ax[r, 0].set_title(f"step {step}", fontsize=8, loc="left")
fig.suptitle("GAN samples from the same noise vectors as training progresses")
fig.tight_layout(); fig.savefig("session-13_gan_samples.png", dpi=100); plt.show()
