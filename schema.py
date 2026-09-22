"""Single source of truth for the v1 parameter set.

v1 scope decided by isolation experiments: only parameters that showed
real learning when their module was varied alone. Everything else is
frozen at Vital's default, which also removes interference.
"""

import random

# Always set, never varied. Switches, routing, and every cut parameter
# at its Vital default value.
FIXED = {
    384: 1.0,       # Oscillator 1 Switch — on
    114: 1.0,       # Filter 1 Switch — on
    106: 1.0,       # Filter 1 FILTER Input — osc 1 routes into filter 1
    393: 0.5,       # Oscillator 1 Tune — cut: below mel resolution
    397: 0.0,       # Oscillator 1 Wave Frame — cut: no audible effect on init table
    386: 0.5,       # Oscillator 1 Phase — cut: phase randomization overrides it
    385: 0.5,       # Oscillator 1 Pan — cut: lost in mono conversion
    395: 0.4472,    # Oscillator 1 Unison Detune — cut: only 1 unison voice
    115: 0.5,       # Filter 1 Resonance — cut: weak, and interferes with cutoff
    105: 0.0,       # Filter 1 Drive — cut
    102: 0.0,       # Filter 1 Blend — cut
    111: 0.5,       # Filter 1 Key Track — cut: needs multiple pitches to be audible
    116: 0.031,     # Filter 1 Style — cut, fixed at 12dB
    51:  0.45,      # Envelope 1 Decay Power — cut
    536: 0.0,       # Envelope 1 Delay — cut
    542: 0.0,       # Envelope 1 Hold — cut
}

# (index, name, lo, hi) — sampled uniformly in [lo, hi]. Regression outputs.
CONTINUOUS = [
    (382, "Oscillator 1 Level",             0.55, 1.00),
    (391, "Oscillator 1 Transpose",         0.00, 1.00),
    (377, "Oscillator 1 Distortion Amount", 0.00, 1.00),
    (104, "Filter 1 Cutoff",                0.15, 1.00),
    (112, "Filter 1 Mix",                   0.00, 1.00),
    (48,  "Envelope 1 Attack",              0.00, 0.70),
    (50,  "Envelope 1 Decay",               0.00, 1.00),
    (54,  "Envelope 1 Sustain",             0.00, 1.00),
    (52,  "Envelope 1 Release",             0.00, 1.00),
    (49,  "Envelope 1 Attack Power",        0.00, 1.00),
    (53,  "Envelope 1 Release Power",       0.00, 1.00),
]

# (index, name, [(raw_value, label), ...]) — measured from the plugin.
# List position is the class index the model predicts.
CATEGORICAL = [
    (113, "Filter 1 Model", [
        (0.03550, "Analog"),
        (0.14275, "Dirty"),
        (0.28575, "Ladder"),
        (0.42850, "Digital"),
        (0.57125, "Diode"),
        (0.71425, "Formant"),
        (0.85725, "Comb"),
        (0.96450, "Phaser"),
    ]),
    (380, "Oscillator 1 Distortion Type", [
        (0.02075, "None"),
        (0.08325, "Sync"),
        (0.16650, "Formant"),
        (0.25000, "Quantize"),
        (0.33325, "Bend"),
        (0.41650, "Squeeze"),
        (0.50000, "Pulse"),
        (0.62500, "FM <- Osc"),
        (0.87500, "RM <- Osc"),
    ]),
]

N_CONTINUOUS = len(CONTINUOUS)
CATEGORICAL_SIZES = [len(opts) for _, _, opts in CATEGORICAL]
CONTINUOUS_NAMES = [n for _, n, _, _ in CONTINUOUS]


def sample_continuous():
    return [lo + random.random() * (hi - lo) for _, _, lo, hi in CONTINUOUS]


def sample_classes():
    return [random.randrange(len(opts)) for _, _, opts in CATEGORICAL]


def apply(synth, continuous, classes):
    for idx, val in FIXED.items():
        synth.set_parameter(idx, val)
    for (idx, _, _, _), val in zip(CONTINUOUS, continuous):
        synth.set_parameter(idx, val)
    for (idx, _, opts), c in zip(CATEGORICAL, classes):
        synth.set_parameter(idx, opts[c][0])