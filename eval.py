"""Per-parameter evaluation of a trained patch matcher.

Usage: python3 eval.py [data_dir]
"""

import numpy as np
import torch
import json, sys
from pathlib import Path

import schema
from train import PatchNet

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

d = np.load(OUT / "features.npz")
X = torch.tensor((d["X"] + 40.0) / 40.0).unsqueeze(1)
y_cont = torch.tensor(d["y_cont"])
y_class = torch.tensor(d["y_class"])

with open(OUT / "labels.json") as f:
    meta = json.load(f)
varied_c = set(meta.get("varied_continuous", schema.CONTINUOUS_NAMES))
varied_k = set(meta.get("varied_categorical", [n for _, n, _ in schema.CATEGORICAL]))

ck = torch.load(OUT / "model.pt", map_location=device)
model = PatchNet(ck["n_cont"], ck["cat_sizes"]).to(device)
model.load_state_dict(ck["state"])
model.eval()

val = ck["val_idx"].cpu()

pcs, pks = [], []
with torch.no_grad():
    for i in range(0, len(val), 128):
        b = val[i:i + 128]
        pc, pk = model(X[b].to(device))
        pcs.append(pc.cpu())
        pks.append(torch.stack([p.argmax(1).cpu() for p in pk], dim=1))

pc, pk = torch.cat(pcs), torch.cat(pks)
yc, yk = y_cont[val], y_class[val]

label = meta.get("group", "full")
print(f"{OUT}  group={label}  val={len(val)} examples")
print("rows marked * were varied; unmarked were frozen\n")
print("continuous — mean absolute error vs chance")

rows = []
for i, (_, name, lo, hi) in enumerate(schema.CONTINUOUS):
    e = (pc[:, i] - yc[:, i]).abs().mean().item()
    chance = (hi - lo) / 3
    rows.append((name in varied_c, e / chance, e, chance, name))

for was_varied, ratio, e, chance, name in sorted(rows, key=lambda r: (not r[0], r[1])):
    if not was_varied:
        print(f"    {e:.3f}  (frozen)                    {name}")
        continue
    tag = "good" if ratio < 0.5 else "weak" if ratio < 0.8 else "none"
    print(f"  * {e:.3f}  chance {chance:.3f}  {ratio:.2f}x  {tag:5s}  {name}")

print("\ncategorical — accuracy")
for j, (_, name, opts) in enumerate(schema.CATEGORICAL):
    acc = (pk[:, j] == yk[:, j]).float().mean().item()
    chance = 1.0 / len(opts)
    mark = "*" if name in varied_k else " "
    if name not in varied_k:
        print(f"    {acc:.3f}  (frozen)                    {name}")
    else:
        print(f"  {mark} {acc:.3f}  chance {chance:.3f}  {acc/chance:.2f}x  {name}")