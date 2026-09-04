import dawdreamer as daw
import csv

engine = daw.RenderEngine(44100, 512)
synth = engine.make_plugin_processor("vital", "/Library/Audio/Plug-Ins/VST3/Vital.vst3")

rows = []
for i in range(synth.get_plugin_parameter_size()):
    rows.append([i, synth.get_parameter_name(i), synth.get_parameter(i)])

with open("params.csv", "w", newline="") as f:
    csv.writer(f).writerows([["index", "name", "default"]] + rows)

print(len(rows), "written")