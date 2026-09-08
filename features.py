"""Precompute mel spectrograms for the dataset."""

import numpy as np
import librosa
import json
from pathlib import Path

import sys
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
N_MELS, N_FFT, HOP = 128, 1024, 256

with open(OUT / "labels.json") as f:
    meta = json.load(f)

sr = meta["sample_rate"]
records = meta["records"]


def spec(path):
    y, _ = librosa.load(path, sr=sr, mono=True)
    m = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT,
                                       hop_length=HOP, n_mels=N_MELS)
    return librosa.power_to_db(m, ref=np.max).astype(np.float32)


first = spec(OUT / "audio" / records[0]["file"])
X = np.empty((len(records), N_MELS, first.shape[1]), dtype=np.float32)
X[0] = first

for i in range(1, len(records)):
    X[i] = spec(OUT / "audio" / records[i]["file"])
    if (i + 1) % 5000 == 0:
        print(f"{i + 1}/{len(records)}")

y_cont = np.array([r["continuous"] for r in records], dtype=np.float32)
y_class = np.array([r["classes"] for r in records], dtype=np.int64)

np.savez(OUT / "features.npz", X=X, y_cont=y_cont, y_class=y_class)
print("X", X.shape, "cont", y_cont.shape, "class", y_class.shape)