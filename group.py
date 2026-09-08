"""Generate a dataset where only one module's parameters vary.

Usage: python3 group.py <filter|envelope|osc> [N]
"""

import dawdreamer as daw
import numpy as np
import soundfile as sf
import json, random, sys
from pathlib import Path

import schema

SR, BUF = 22050, 512
PLUGIN = "/Library/Audio/Plug-Ins/VST3/Vital.vst3"

GROUPS = {
    "filter": {
        "continuous": ["Filter 1 Cutoff", "Filter 1 Resonance", "Filter 1 Drive",
                       "Filter 1 Blend", "Filter 1 Mix", "Filter 1 Key Track"],
        "categorical": ["Filter 1 Model", "Filter 1 Style"],
    },
    "envelope": {
        "continuous": ["Envelope 1 Attack", "Envelope 1 Decay", "Envelope 1 Sustain",
                       "Envelope 1 Release", "Envelope 1 Attack Power",
                       "Envelope 1 Decay Power", "Envelope 1 Release Power",
                       "Envelope 1 Delay", "Envelope 1 Hold"],
        "categorical": [],
    },
    "osc": {
        "continuous": ["Oscillator 1 Level", "Oscillator 1 Transpose", "Oscillator 1 Tune",
                       "Oscillator 1 Wave Frame", "Oscillator 1 Phase", "Oscillator 1 Pan",
                       "Oscillator 1 Unison Detune", "Oscillator 1 Distortion Amount"],
        "categorical": ["Oscillator 1 Distortion Type"],
    },
}

name = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
group = GROUPS[name]

cont_names = schema.CONTINUOUS_NAMES
cat_names = [n for _, n, _ in schema.CATEGORICAL]
vary_cont = [cont_names.index(n) for n in group["continuous"]]
vary_cat = [cat_names.index(n) for n in group["categorical"]]

base = [(lo + hi) / 2 for _, _, lo, hi in schema.CONTINUOUS]

OUT = Path(f"data_{name}")
(OUT / "audio").mkdir(parents=True, exist_ok=True)

engine = daw.RenderEngine(SR, BUF)
synth = engine.make_plugin_processor("vital", PLUGIN)

records = []
for i in range(N):
    cont = list(base)
    for j in vary_cont:
        lo, hi = schema.CONTINUOUS[j][2], schema.CONTINUOUS[j][3]
        cont[j] = lo + random.random() * (hi - lo)

    classes = [0] * len(schema.CATEGORICAL)
    for j in vary_cat:
        classes[j] = random.randrange(len(schema.CATEGORICAL[j][2]))

    synth.clear_midi()
    schema.apply(synth, cont, classes)
    synth.add_midi_note(60, 100, 0.0, 2.0)
    engine.load_graph([(synth, [])])
    engine.render(3.0)
    audio = engine.get_audio().mean(axis=0)

    if float(np.abs(audio).max()) < 0.001:
        continue

    fname = f"{i:06d}.wav"
    sf.write(OUT / "audio" / fname, audio, SR)
    records.append({"file": fname, "continuous": cont, "classes": classes})

    if (i + 1) % 1000 == 0:
        print(f"{i + 1}/{N}  kept {len(records)}")

with open(OUT / "labels.json", "w") as f:
    json.dump({"sample_rate": SR, "categorical_sizes": schema.CATEGORICAL_SIZES,
               "group": name, "varied_continuous": group["continuous"],
               "varied_categorical": group["categorical"], "records": records}, f)

print(f"done: {len(records)} kept of {N} -> {OUT}")