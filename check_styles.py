import dawdreamer as daw
import numpy as np

engine = daw.RenderEngine(44100, 512)
synth = engine.make_plugin_processor("vital", "/Library/Audio/Plug-Ins/VST3/Vital.vst3")

for model in range(8):
    synth.set_parameter(113, model / 7.0)
    styles = []
    for v in np.linspace(0, 1, 101):
        synth.set_parameter(116, float(v))
        t = synth.get_parameter_text(116)
        if t not in styles:
            styles.append(t)
    print(f"{synth.get_parameter_text(113)}: {styles}")