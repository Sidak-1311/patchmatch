import dawdreamer as daw
import schema

engine = daw.RenderEngine(44100, 512)
synth = engine.make_plugin_processor("vital", "/Library/Audio/Plug-Ins/VST3/Vital.vst3")

for idx, name, n, total in schema.CATEGORICAL:
    print(name)
    for c in range(n):
        synth.set_parameter(idx, schema.categorical_to_raw(c, total))
        print(" ", c, synth.get_parameter_text(idx))