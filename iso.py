"""Generate a dataset where only one parameter varies."""

import dawdreamer as daw
import numpy as np
import soundfile as sf
import json, sys
from pathlib import Path

import schema

SR, BUF = 22050, 512
PLUGIN = "/Library/Audio/Plug-Ins/VST3/Vital.vst3"
TARGET = "Filter 1 Cutoff"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5000

OUT = Path("data_iso")
(OUT / "audio").mkdir(parents=True, exist_ok=True)

names = schema.CONTINUOUS_NAMES
ti = names.index(TARGET)
lo, hi = schema.CONTINUOUS[ti][2], schema.CONTINUOUS[ti][3]

# every continuous param at midpoint, every categorical at class 0
base = [(l + h) / 2 for _, _, l, h in schema.CONTINUOUS]
classes = [0 for _ in schema.CATEGORICAL]

engine = daw.RenderEngine(SR, BUF)
synth = engine.make_plugin_processor("vital", PLUGIN)

records = []
for i in range(N):
    cont = list(base)
    cont[ti] = lo + (hi - lo) * (i / (N - 1))

    synth.clear_midi()
    schema.apply(synth, cont, classes)
    synth.add_midi_note(60, 100, 0.0, 2.0)
    engine.load_graph([(synth, [])])
    engine.render(3.0)
    audio = engine.get_audio().mean(axis=0)

    if float(np.abs(audio).max()) < 0.001:
        continue

    name = f"{i:06d}.wav"
    sf.write(OUT / "audio" / name, audio, SR)
    records.append({"file": name, "continuous": cont, "classes": classes})

    if (i + 1) % 1000 == 0:
        print(f"{i + 1}/{N}")

with open(OUT / "labels.json", "w") as f:
    json.dump({"sample_rate": SR, "categorical_sizes": schema.CATEGORICAL_SIZES,
               "target": TARGET, "records": records}, f)

print(f"done: {len(records)} kept")