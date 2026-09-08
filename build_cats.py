import dawdreamer as daw
import numpy as np

engine = daw.RenderEngine(44100, 512)
synth = engine.make_plugin_processor("vital", "/Library/Audio/Plug-Ins/VST3/Vital.vst3")

for idx in [113, 116, 380]:
    vs = np.linspace(0, 1, 2001)
    labels = []
    for v in vs:
        synth.set_parameter(idx, float(v))
        labels.append(synth.get_parameter_text(idx))

    runs, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            mid = (vs[start] + vs[i - 1]) / 2
            runs.append((labels[start], round(float(mid), 5)))
            start = i

    print(f"{idx} -> [")
    for label, v in runs:
        print(f"    ({v}, {label!r}),")
    print("]")