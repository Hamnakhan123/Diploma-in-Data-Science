"""Content vs style: feature maps differ when layout changes, Gram matrices (texture statistics) stay similar."""
import numpy as np
import torch
from torch import nn
torch.manual_seed(0)

def stripes(h=32, w=32, angle=0.0, freq=6, shift=0):
    yy, xx = np.mgrid[0:h, 0:w]
    return np.sin(freq * 2 * np.pi * (np.cos(angle) * (xx + shift) + np.sin(angle) * yy) / w)

def dots(h=32, w=32, shift=0):
    yy, xx = np.mgrid[0:h, 0:w]
    return np.sin(2 * np.pi * (xx + shift) / 8) * np.sin(2 * np.pi * yy / 8)

imgs = {"stripes": stripes(), "stripes moved": stripes(shift=3), "dots": dots()}
cnn = nn.Sequential(nn.Conv2d(1, 16, 5, padding=2), nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1), nn.ReLU())

def features(img):
    with torch.no_grad():
        return cnn(torch.tensor(img, dtype=torch.float32)[None, None])[0]        # (C, H, W)

def gram(F):
    C = F.shape[0]; Fm = F.reshape(C, -1)
    return Fm @ Fm.T / Fm.shape[1]                                                # (C, C): which features co-occur

feats = {k: features(v) for k, v in imgs.items()}
grams = {k: gram(f) for k, f in feats.items()}
def dist(a, b): return (a - b).pow(2).mean().sqrt().item()
print(f"{'pair':<28}{'content distance':>18}{'style (Gram) distance':>24}")
for a, b in [("stripes", "stripes moved"), ("stripes", "dots")]:
    print(f"{a + ' vs ' + b:<28}{dist(feats[a], feats[b]):>18.4f}{dist(grams[a], grams[b]):>24.4f}")
