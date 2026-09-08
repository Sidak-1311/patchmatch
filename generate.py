"""Generate a training dataset: random Vital patches -> audio + labels.

Usage: python3 generate.py [N]
"""

import dawdreamer as daw
import numpy as np
import soundfile as sf
import json, sys, time
from pathlib import Path

import schema

SR, BUF = 22050, 512
PLUGIN = "/Library/Audio/Plug-Ins/VST3/Vital.vst3"
NOTE, VELOCITY, NOTE_LEN, RENDER_LEN = 60, 100, 2.0, 3.0
SILENCE_THRESHOLD = 0.001

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
OUT = Path("data")
(OUT / "audio").mkdir(parents=True, exist_ok=True)

engine = daw.RenderEngine(SR, BUF)
synth = engine.make_plugin_processor("vital", PLUGIN)

records = []
start = time.time()

for i in range(N):
    continuous = schema.sample_continuous()
    classes = schema.sample_classes()

    synth.clear_midi()
    schema.apply(synth, continuous, classes)
    synth.add_midi_note(NOTE, VELOCITY, 0.0, NOTE_LEN)
    engine.load_graph([(synth, [])])
    engine.render(RENDER_LEN)

    audio = engine.get_audio().mean(axis=0)

    if float(np.abs(audio).max()) < SILENCE_THRESHOLD:
        continue

    name = f"{i:06d}.wav"
    sf.write(OUT / "audio" / name, audio, SR)
    records.append({"file": name, "continuous": continuous, "classes": classes})

    if (i + 1) % 1000 == 0:
        rate = (i + 1) / (time.time() - start)
        print(f"{i + 1}/{N}  kept {len(records)}  {rate:.0f}/s")

meta = {
    "sample_rate": SR,
    "render_length": RENDER_LEN,
    "note": NOTE,
    "velocity": VELOCITY,
    "note_length": NOTE_LEN,
    "continuous_names": schema.CONTINUOUS_NAMES,
    "categorical_names": [n for _, n, _ in schema.CATEGORICAL],
    "categorical_sizes": schema.CATEGORICAL_SIZES,
    "records": records,
}

with open(OUT / "labels.json", "w") as f:
    json.dump(meta, f)

print(f"done: {len(records)} kept of {N} in {time.time() - start:.0f}s")