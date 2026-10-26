"""A PyTorch multilayer perceptron for breast-cancer diagnosis: DataLoader, training loop, early stopping."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
torch.manual_seed(0); torch.set_num_threads(2)

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.2, random_state=0, stratify=y)
X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=.2, random_state=0, stratify=y_tr)
sc = StandardScaler().fit(X_tr)
t = lambda a: torch.tensor(a, dtype=torch.float32)
X_tr, X_va, X_te = t(sc.transform(X_tr)), t(sc.transform(X_va)), t(sc.transform(X_te))
y_tr, y_va, y_te = t(y_tr).unsqueeze(1), t(y_va).unsqueeze(1), t(y_te).unsqueeze(1)
loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)

model = nn.Sequential(nn.Linear(30, 32), nn.ReLU(), nn.Dropout(0.2),
                      nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))      # outputs a logit
print(model)
print("trainable parameters:", sum(p.numel() for p in model.parameters()))
loss_fn = nn.BCEWithLogitsLoss()                                         # sigmoid + binary cross-entropy
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

best, best_state, patience, wait = np.inf, None, 15, 0
for epoch in range(300):
    model.train()
    for xb, yb in loader:
        opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        val = loss_fn(model(X_va), y_va).item()
    if val < best - 1e-4:
        best, best_state, wait = val, {k: v.clone() for k, v in model.state_dict().items()}, 0
    else:
        wait += 1
    if epoch % 20 == 0:
        print(f"epoch {epoch:>3}: train loss {loss.item():.4f}  val loss {val:.4f}")
    if wait >= patience:
        print(f"early stopping at epoch {epoch} (best val loss {best:.4f})"); break

model.load_state_dict(best_state); model.eval()
with torch.no_grad():
    acc = ((model(X_te) > 0).float() == y_te).float().mean().item()
print(f"test accuracy: {acc:.3f}")
