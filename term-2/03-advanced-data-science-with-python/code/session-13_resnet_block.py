"""Plain deep MLP vs the same depth with residual connections: gradient flow and training loss."""
import torch
from torch import nn
torch.set_num_threads(2)

class Block(nn.Module):
    def __init__(self, width, residual):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(width, width), nn.Tanh(), nn.Linear(width, width))
        self.residual = residual
    def forward(self, x):
        return torch.tanh(self.f(x) + x) if self.residual else torch.tanh(self.f(x))

def make(depth, residual):
    return nn.Sequential(nn.Linear(10, 32), *[Block(32, residual) for _ in range(depth)], nn.Linear(32, 1))

torch.manual_seed(0)
X = torch.randn(512, 10); y = (X[:, :3].sum(1, keepdim=True) > 0).float()
print("depth  type       grad norm at first layer   loss after 200 steps")
for depth in [2, 20, 40]:
    for residual in [False, True]:
        torch.manual_seed(0)
        m = make(depth, residual)
        loss = nn.functional.binary_cross_entropy_with_logits(m(X), y); loss.backward()
        g = m[0].weight.grad.norm().item()
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        for _ in range(200):
            opt.zero_grad(); l = nn.functional.binary_cross_entropy_with_logits(m(X), y); l.backward(); opt.step()
        print(f"{depth:>5}  {'residual' if residual else 'plain':<9}  {g:>24.2e}   {l.item():>20.3f}")
