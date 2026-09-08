import dawdreamer as daw
import numpy as np

engine = daw.RenderEngine(44100, 512)
synth = engine.make_plugin_processor("vital", "/Library/Audio/Plug-Ins/VST3/Vital.vst3")

for idx in [113, 116, 380]:
    seen = []
    for v in np.linspace(0, 1, 101):
        synth.set_parameter(idx, float(v))
        t = synth.get_parameter_text(idx)
        if t not in seen:
            seen.append(t)
    print(f"{idx}: {len(seen)} options -> {seen}")