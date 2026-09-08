# Patch Matcher

ML system: audio in → Vital synth parameters out.
Python + PyTorch + DawDreamer. macOS arm64, Python 3.14, venv at .venv.

## Key facts about Vital's parameter space
- get_plugin_parameter_size() returns 2983. Most is noise:
  - 2080 = MIDI CC wrapper slots (JUCE artifact, 128 CCs x 16 channels). Ignore.
  - 448 = modulation matrix. Out of scope for v1.
  - ~455 = actual synth parameters.
- Parameter names from the VST3 are GUI display names ("Filter 1 Cutoff"),
  NOT the snake_case keys in .vital preset JSON. That mapping is unsolved.
- The list is ordered alphabetically, so module params are not contiguous.
  Select by name match, never by index range.

## v1 scope (locked)
Oscillator 1 + Filter 1 + Envelope 1 only, ~30 parameters.
Everything else stays at default.
The wavetable must stay FIXED — it's embedded preset data, not a parameter,
so Wave Frame is meaningless if the table varies between examples.

Categoricals (need classification heads, not regression):
Filter Model, Filter Style, Distortion Type.

## Constraints
- Import dawdreamer BEFORE torch. Both embed LLVM; wrong order segfaults.
  Better: keep data generation and training in separate scripts entirely.
- Use python3/pip3, not python/pip.
- Verify osc→filter routing (index 106) audibly before generating any dataset.

## Next step
Write schema.py as the single source of truth for the v1 parameter set.