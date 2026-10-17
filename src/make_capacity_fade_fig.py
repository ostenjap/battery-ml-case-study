"""Regenerates the actual-vs-predicted capacity fade curves for the report
(Section 3.2), reusing the exact model definitions/hyperparameters from
documents/step3_baseline_summary.ipynb: MLP (tanh), MLP (ReLU), LSTM, SEI PINN.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "processed" / "clean_data.csv"
OUT = REPO / "reports" / "figures" / "capacity_fade_curves.png"

df = pd.read_csv(DATA, sep=";").sort_values("cycle").reset_index(drop=True)
train_df = df[df.cycle <= 90].reset_index(drop=True)
test_df = df[df.cycle > 90].reset_index(drop=True)
n_max = train_df.cycle.max()

x_train = torch.tensor(train_df.cycle.values / n_max, dtype=torch.float32).reshape(-1, 1)
y_train = torch.tensor(train_df.capacity_ah.values, dtype=torch.float32).reshape(-1, 1)
x_all = torch.tensor(df.cycle.values / n_max, dtype=torch.float32).reshape(-1, 1)

curves = {}

for name, activation in [("MLP (tanh)", nn.Tanh), ("MLP (ReLU)", nn.ReLU)]:
    torch.manual_seed(0)
    net = nn.Sequential(
        nn.Linear(1, 32), activation(),
        nn.Linear(32, 32), activation(),
        nn.Linear(32, 32), activation(),
        nn.Linear(32, 1),
    )
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    for epoch in range(3000):
        opt.zero_grad()
        loss = torch.mean((net(x_train) - y_train) ** 2)
        loss.backward()
        opt.step()
    with torch.no_grad():
        curves[name] = net(x_all).squeeze().numpy()

WINDOW = 5
cap_train = train_df.capacity_ah.values.astype(np.float32)

X = torch.tensor(np.array([cap_train[i:i + WINDOW] for i in range(len(cap_train) - WINDOW)])).unsqueeze(-1)
Y = torch.tensor(np.array([cap_train[i + WINDOW] for i in range(len(cap_train) - WINDOW)])).unsqueeze(-1)


class LSTMNet(nn.Module):
    def __init__(self, hidden=32, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


# GPU when available, exactly as step2 does -- the reported LSTM row is the
# GPU one, and cuDNN's kernels do not match CPU bit-for-bit, so the curve has
# to be produced on the same device as the metric it illustrates.
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(0)
lstm = LSTMNet().to(device)
Xd, Yd = X.to(device), Y.to(device)
opt = torch.optim.Adam(lstm.parameters(), lr=5e-3)
for epoch in range(1500):
    opt.zero_grad()
    loss = torch.mean((lstm(Xd) - Yd) ** 2)
    loss.backward()
    opt.step()

lstm.eval()
with torch.no_grad():
    # teacher-forced one-step predictions over the training windows (matches
    # the (X, Y) pairs the LSTM was trained on -- no cycles 2..(2+WINDOW-1),
    # since those are only ever seen as input, never as a predicted target)
    lstm_train_curve = lstm(Xd).squeeze().cpu().numpy()

    # recursive forecast over the held-out region, identical to step2's loop
    window, lstm_test_curve = list(cap_train[-WINDOW:]), []
    for _ in range(len(test_df)):
        inp = torch.tensor(window[-WINDOW:], dtype=torch.float32).reshape(1, WINDOW, 1).to(device)
        nxt = lstm(inp).item()
        lstm_test_curve.append(nxt)
        window.append(nxt)
    lstm_test_curve = np.array(lstm_test_curve)

# sanity: this must reproduce step2's reported held-out error
_rmse = float(np.sqrt(np.mean((lstm_test_curve - test_df.capacity_ah.values) ** 2)))
_mae = float(np.mean(np.abs(lstm_test_curve - test_df.capacity_ah.values)))
print(f"LSTM ({device}) held-out RMSE {_rmse:.4f}  MAE {_mae:.4f}   (step2: 0.0363 / 0.0291)")

lstm_cycles = np.concatenate([train_df.cycle.values[WINDOW:], test_df.cycle.values])
lstm_curve = np.concatenate([lstm_train_curve, lstm_test_curve])

Q0_AH = 1.855
tau_col = torch.linspace(0, 1, 256).reshape(-1, 1)


class SEIPinn(nn.Module):
    """step4's network: 4 hidden layers, three tanh then one sigmoid."""

    def __init__(self, hidden=32, delta0=1.0):
        super().__init__()
        self.delta0 = delta0
        activations = [nn.Tanh, nn.Tanh, nn.Tanh, nn.Sigmoid]
        layers = [nn.Linear(1, hidden), activations[0]()]
        for act in activations[1:]:
            layers += [nn.Linear(hidden, hidden), act()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)
        self.log_k = nn.Parameter(torch.tensor(math.log(1e-3)))
        self.log_c_li = nn.Parameter(torch.tensor(math.log(0.1)))

    @property
    def k(self):
        return torch.exp(self.log_k)

    @property
    def c_li(self):
        return torch.exp(self.log_c_li)

    def u(self, tau):
        return self.delta0 ** 2 + tau * F.softplus(self.net(tau))

    def capacity(self, tau):
        delta = torch.sqrt(self.u(tau))
        return Q0_AH - self.c_li * (delta - self.delta0)


def physics_residual(model, tau):
    tau = tau.requires_grad_(True)
    u = model.u(tau)
    du_dtau = torch.autograd.grad(u, tau, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    du_dn = du_dtau / n_max
    return du_dn - 2.0 * model.k


torch.manual_seed(0)
pinn = SEIPinn()
opt = torch.optim.Adam(pinn.parameters(), lr=5e-3)
lambda_phys = 100.0
for epoch in range(3000):
    opt.zero_grad()
    loss_data = torch.mean((pinn.capacity(x_train) - y_train) ** 2)
    loss_phys = torch.mean(physics_residual(pinn, tau_col.clone()) ** 2)
    loss = loss_data + lambda_phys * loss_phys
    loss.backward()
    opt.step()
pinn.eval()
with torch.no_grad():
    curves["SEI PINN"] = pinn.capacity(x_all).squeeze().numpy()

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(df.cycle, df.capacity_ah, "o", color="black", ms=3, label="Actual", zorder=5)
colors = {"MLP (tanh)": "tab:red", "MLP (ReLU)": "tab:orange", "SEI PINN": "tab:green"}
for name, curve in curves.items():
    ax.plot(df.cycle, curve, color=colors[name], label=name, linewidth=1.8)
# LSTM: teacher-forced one-step predictions over training windows (cycles
# 2..2+WINDOW-1 have no prediction, only ever used as input), then the same
# recursive held-out forecast used for its Table 1 metrics.
ax.plot(lstm_cycles, lstm_curve, color="tab:blue", label="LSTM", linewidth=1.8)
ax.axvline(90.5, color="grey", linestyle="--", linewidth=1)
ax.text(90.5, ax.get_ylim()[1], " held-out ->", va="top", ha="left", fontsize=8, color="grey")
ax.set_xlabel("Cycle")
ax.set_ylabel("Capacity (Ah)")
ax.set_title("Actual vs. predicted capacity fade")
ax.legend()
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200)
print(f"wrote {OUT}")
