"""Run the trained model on a sound and render its guess.

Usage:
  python3 infer.py            # random patch, so you know the true answer
  python3 infer.py sound.wav  # your own audio
"""

import dawdreamer as daw          # must come before torch (LLVM conflict)
import numpy as np
import soundfile as sf
import librosa
import json, sys
from pathlib import Path

import torch
import schema
from train import PatchNet

DATA = Path("data")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

PLUGIN = "/Library/Audio/Plug-Ins/VST3/Vital.vst3"
BUF = 512
NOTE, VELOCITY, NOTE_LEN, RENDER_LEN = 60, 100, 2.0, 3.0
N_MELS, N_FFT, HOP = 128, 1024, 256      # must match features.py

with open(DATA / "labels.json") as f:
    SR = json.load(f)["sample_rate"]

ck = torch.load(DATA / "model.pt", map_location="cpu")
if ck["n_cont"] != schema.N_CONTINUOUS:
    sys.exit("model.pt doesn't match schema.py — retrain or restore the schema")
model = PatchNet(ck["n_cont"], ck["cat_sizes"])
model.load_state_dict(ck["state"])
model.eval()


def render(cont, classes):
    """Fresh engine each call so no plugin state carries between renders."""
    eng = daw.RenderEngine(SR, BUF)
    s = eng.make_plugin_processor("vital", PLUGIN)
    schema.apply(s, cont, classes)
    s.add_midi_note(NOTE, VELOCITY, 0.0, NOTE_LEN)
    eng.load_graph([(s, [])])
    eng.render(RENDER_LEN)
    return eng.get_audio().mean(axis=0)


def spectrogram(audio):
    n = int(SR * RENDER_LEN)
    audio = np.pad(audio, (0, max(0, n - len(audio))))[:n]
    m = librosa.feature.melspectrogram(y=audio, sr=SR, n_fft=N_FFT,
                                       hop_length=HOP, n_mels=N_MELS)
    s = librosa.power_to_db(m, ref=np.max).astype(np.float32)
    return torch.tensor((s + 40.0) / 40.0)[None, None]


def predict(audio):
    with torch.no_grad():
        pc, pk = model(spectrogram(audio))
    return pc[0].tolist(), [int(p.argmax(1)) for p in pk]


def distance(a, b):
    return (spectrogram(a) - spectrogram(b)).abs().mean().item()


# --- target sound ---
if len(sys.argv) > 1:
    target, _ = librosa.load(sys.argv[1], sr=SR, mono=True)
    true_c = true_k = None
else:
    while True:
        true_c, true_k = schema.sample_continuous(), schema.sample_classes()
        target = render(true_c, true_k)
        if np.abs(target).max() > 0.001:
            break

# --- the model's guess ---
pred_c, pred_k = predict(target)
guess = render(pred_c, pred_k)

# a random patch, as a baseline for "how close is close"
rand = render(schema.sample_continuous(), schema.sample_classes())

sf.write(OUT / "target.wav", target, SR)
sf.write(OUT / "guess.wav", guess, SR)

print(f"\n{'parameter':34s} {'true':>6s} {'guess':>6s}")
for i, name in enumerate(schema.CONTINUOUS_NAMES):
    t = f"{true_c[i]:.2f}" if true_c else "-"
    print(f"{name:34s} {t:>6s} {pred_c[i]:6.2f}")
for j, (_, name, opts) in enumerate(schema.CATEGORICAL):
    t = opts[true_k[j]][1] if true_k else "-"
    print(f"{name:34s} {t:>10s} -> {opts[pred_k[j]][1]}")

print(f"\npeaks   target {np.abs(target).max():.3f}"
      f"   guess {np.abs(guess).max():.3f}"
      f"   random {np.abs(rand).max():.3f}")
print(f"spectrogram distance   guess: {distance(target, guess):.3f}"
      f"   random patch: {distance(target, rand):.3f}")
print(f"saved {OUT}/target.wav and {OUT}/guess.wav")