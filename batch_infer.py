"""Run inference N times and save a summary.

Usage: python3 batch_infer.py [N]
"""

import numpy as np
import soundfile as sf
import sys, json
from pathlib import Path

from infer import (render, predict, distance, schema, SR, OUT)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
RUNS = Path("runs")
RUNS.mkdir(exist_ok=True)

lines, rows = [], []

for run in range(N):
    while True:
        true_c, true_k = schema.sample_continuous(), schema.sample_classes()
        target = render(true_c, true_k)
        if np.abs(target).max() > 0.001:
            break

    pred_c, pred_k = predict(target)
    guess = render(pred_c, pred_k)

    sf.write(RUNS / f"{run:02d}_target.wav", target, SR)
    sf.write(RUNS / f"{run:02d}_guess.wav", guess, SR)

    rows.append({"true_c": true_c, "pred_c": pred_c,
                 "true_k": true_k, "pred_k": pred_k})

    lines.append(f"\n=== run {run} ===")
    lines.append(f"{'parameter':34s} {'true':>6s} {'guess':>6s} {'err':>6s}")
    for i, name in enumerate(schema.CONTINUOUS_NAMES):
        e = abs(pred_c[i] - true_c[i])
        lines.append(f"{name:34s} {true_c[i]:6.2f} {pred_c[i]:6.2f} {e:6.2f}")
    for j, (_, name, opts) in enumerate(schema.CATEGORICAL):
        hit = "ok" if pred_k[j] == true_k[j] else "MISS"
        lines.append(f"{name:34s} {opts[true_k[j]][1]:>10s} -> "
                     f"{opts[pred_k[j]][1]:10s} {hit}")
    lines.append(f"peaks target {np.abs(target).max():.3f} "
                 f"guess {np.abs(guess).max():.3f}   "
                 f"distance {distance(target, guess):.3f}")
    print(f"run {run} done")

# --- averages across all runs ---
lines.append(f"\n=== mean absolute error over {N} runs ===")
for i, name in enumerate(schema.CONTINUOUS_NAMES):
    errs = [abs(r["pred_c"][i] - r["true_c"][i]) for r in rows]
    lines.append(f"{name:34s} {np.mean(errs):6.3f}   max {max(errs):.3f}")
for j, (_, name, _) in enumerate(schema.CATEGORICAL):
    acc = np.mean([r["pred_k"][j] == r["true_k"][j] for r in rows])
    lines.append(f"{name:34s} {acc:6.2f} accuracy")

text = "\n".join(lines)
(RUNS / "summary.txt").write_text(text)
with open(RUNS / "runs.json", "w") as f:
    json.dump(rows, f)

print(text)
print(f"\nsaved to {RUNS}/")