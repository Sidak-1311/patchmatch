"""Train the patch matcher."""

import numpy as np
import torch
import torch.nn as nn
import json
from pathlib import Path

import sys
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
EPOCHS, BATCH, LR, CAT_WEIGHT = 20, 64, 3e-4, 0.1


class PatchNet(nn.Module):
    def __init__(self, n_cont, cat_sizes):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
        )
        self.cont = nn.Linear(128 * 16, n_cont)
        self.cats = nn.ModuleList([nn.Linear(128 * 16, n) for n in cat_sizes])

    def forward(self, x):
        h = self.body(x)
        return torch.sigmoid(self.cont(h)), [c(h) for c in self.cats]


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)

    d = np.load(OUT / "features.npz")
    X = torch.tensor((d["X"] + 40.0) / 40.0).unsqueeze(1)
    y_cont = torch.tensor(d["y_cont"])
    y_class = torch.tensor(d["y_class"])

    with open(OUT / "labels.json") as f:
        cat_sizes = json.load(f)["categorical_sizes"]

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(X), generator=g)
    n_val = len(X) // 10
    val, tr = perm[:n_val], perm[n_val:]

    model = PatchNet(y_cont.shape[1], cat_sizes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    mse, ce = nn.MSELoss(), nn.CrossEntropyLoss()

    def run(idx, train):
        model.train(train)
        tc = tk = n = 0.0
        order = idx[torch.randperm(len(idx))] if train else idx
        for i in range(0, len(order), BATCH):
            b = order[i:i + BATCH]
            xb, yc, yk = X[b].to(device), y_cont[b].to(device), y_class[b].to(device)
            with torch.set_grad_enabled(train):
                pc, pk = model(xb)
                loss_cont = mse(pc, yc)
                loss_cat = sum(ce(lg, yk[:, j]) for j, lg in enumerate(pk))
                loss = loss_cont + CAT_WEIGHT * loss_cat
                if train:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            tc += loss_cont.item() * len(b)
            tk += loss_cat.item() * len(b)
            n += len(b)
        return tc / n, tk / n

    best = float("inf")
    for ep in range(EPOCHS):
        trc, trk = run(tr, True)
        vac, vak = run(val, False)
        total = vac + CAT_WEIGHT * vak
        mark = ""
        if total < best:
            best = total
            torch.save({"state": model.state_dict(), "cat_sizes": cat_sizes,
                        "n_cont": y_cont.shape[1], "val_idx": val},
                       OUT / "model.pt")
            mark = "  *"
        print(f"epoch {ep+1:3d}  train cont {trc:.4f} cat {trk:.3f}  "
              f"val cont {vac:.4f} cat {vak:.3f}{mark}")

    print(f"best val total {best:.4f}")