"""Why LSTMs matter: how much does the FIRST input still influence the last hidden state? Then an LSTM forecaster."""
import numpy as np
import torch
from torch import nn
torch.set_num_threads(2)

def gradient_to_first_input(cell, length, trials=20):
    """Average |d h_last / d x_0| over random initialisations: the signal that reaches step 0 in backprop."""
    vals = []
    for k in range(trials):
        torch.manual_seed(k)
        rnn = {"RNN": nn.RNN, "LSTM": nn.LSTM}[cell](1, 32, batch_first=True)
        if cell == "LSTM":                       # common practice: forget-gate bias = 1 (remember by default)
            with torch.no_grad():
                for name, p in rnn.named_parameters():
                    if "bias_ih" in name:
                        p[32:64] = 1.0
        x = torch.randn(1, length, 1, requires_grad=True)
        h, _ = rnn(x)
        h[0, -1].sum().backward()
        vals.append(x.grad[0, 0, 0].abs().item())
    return np.mean(vals)

print("sequence length   gradient reaching step 0:  plain RNN      LSTM")
for length in [5, 20, 50, 100]:
    print(f"{length:>15}   {'':>24}{gradient_to_first_input('RNN', length):9.1e}{gradient_to_first_input('LSTM', length):10.1e}")

# Many-to-one forecasting: predict the next value from the previous 24
t = np.arange(1500)
series = np.sin(2 * np.pi * t / 50) + 0.5 * np.sin(2 * np.pi * t / 13) + np.random.default_rng(0).normal(0, .1, 1500)
W = 24
Xs = torch.tensor(np.stack([series[i:i + W] for i in range(len(series) - W)]), dtype=torch.float32).unsqueeze(2)
ys = torch.tensor(series[W:], dtype=torch.float32)
split = 1200
lstm = nn.LSTM(1, 32, batch_first=True); head = nn.Linear(32, 1)
opt = torch.optim.Adam(list(lstm.parameters()) + list(head.parameters()), lr=5e-3)
for epoch in range(20):
    for i in range(0, split, 64):
        h, _ = lstm(Xs[i:i + 64]); pred = head(h[:, -1]).squeeze(1)
        opt.zero_grad(); loss = nn.functional.mse_loss(pred, ys[i:i + 64]); loss.backward(); opt.step()
with torch.no_grad():
    h, _ = lstm(Xs[split:]); pred = head(h[:, -1]).squeeze(1)
mae = (pred - ys[split:]).abs().mean().item()
naive = (Xs[split:, -1, 0] - ys[split:]).abs().mean().item()
print(f"\nLSTM next-step forecast MAE on unseen data: {mae:.3f}  (naive 'same as last value': {naive:.3f})")
