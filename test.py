import dawdreamer as daw
import numpy as np
import random
import schema

engine = daw.RenderEngine(44100, 512)
synth = engine.make_plugin_processor("vital", "/Library/Audio/Plug-Ins/VST3/Vital.vst3")

for i in range(20):
    cont = [random.random() for _ in schema.CONTINUOUS]
    cats = [random.randrange(len(o)) for _, _, o in schema.CATEGORICAL]

    synth.clear_midi()
    for idx, val in schema.FIXED.items():
        synth.set_parameter(idx, val)
    for (idx, _), val in zip(schema.CONTINUOUS, cont):
        synth.set_parameter(idx, val)
    for (idx, _, opts), c in zip(schema.CATEGORICAL, cats):
        synth.set_parameter(idx, opts[c][0])

    synth.add_midi_note(60, 100, 0.0, 2.0)
    engine.load_graph([(synth, [])])
    engine.render(3.0)
    peak = float(np.abs(engine.get_audio()).max())

    d = dict(zip([n for _, n in schema.CONTINUOUS], cont))
    print(f"peak {peak:.4f}  lvl {d['Oscillator 1 Level']:.2f}"
          f"  cut {d['Filter 1 Cutoff']:.2f}"
          f"  mix {d['Filter 1 Mix']:.2f}"
          f"  atk {d['Envelope 1 Attack']:.2f}"
          f"  sus {d['Envelope 1 Sustain']:.2f}"
          f"  dly {d['Envelope 1 Delay']:.2f}"
          f"  hold {d['Envelope 1 Hold']:.2f}"
          f"  {schema.CATEGORICAL[0][2][cats[0]][1]}")